# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep GPU SM clocks and benchmark a CuTeDSL kernel at each clock, collecting NVML
clock/power telemetry throughout.

The kernel is compiled exactly once; for each locked clock value, the already-compiled
kernel is re-benchmarked via `cutlass.testing.benchmark`, sandwiched by an NVML sampler
so only the timed region contributes to the reported average clock/power.

Currently wired up for dense_gemm_persistent.py, but the driver (calibration, clock
locking, NVML sampling, table/CSV output) is generic: any kernel module that exposes
`prepare_parser()` and a `run(..., benchmark_sweep=...)` hook accepting a `BenchmarkContext`
(see dense_gemm/dense_gemm_persistent.py) can be swept by pointing `--module` at it.

Example:
    python sweep_clocks.py \\
        --clock-min 1200 --clock-max 1900 --clock-step 50 --target-seconds 15 \\
        -- \\
        --mnkl=4096,13568,16384,1 --use_cold_l2 --init_normal --normal_mean=0.0 \\
        --normal_std=0.1 --a_dtype=BFloat16 --b_dtype=BFloat16 --c_dtype=BFloat16 \\
        --acc_dtype=Float32 --a_major=k --b_major=k --c_major=m --use_tma_store \\
        --cluster_shape_mn=2,1 --use_2cta_instrs --mma_tiler=256,256,64 \\
        --mma_inst_shape=256,256,16 --swizzle_size=1 --raster_order=m
"""

import argparse
import csv
import importlib
import os
import sys
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.dirname(_this_dir))  # kernel/, so "dense_gemm.foo" imports resolve

from gpu_clock import lock_sm_clock, reset_sm_clock
from nvml_sampler import NvmlSampler

from cutlass import testing


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep GPU SM clocks while benchmarking a CuTeDSL kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--module",
        type=str,
        default="dense_gemm.dense_gemm_persistent",
        help="Dotted module path of the kernel script to benchmark. Must expose "
        "prepare_parser() and run(..., benchmark_sweep=...).",
    )
    parser.add_argument("--clock-min", type=int, default=1200, help="Lowest SM clock (MHz)")
    parser.add_argument("--clock-max", type=int, default=2400, help="Highest SM clock (MHz)")
    parser.add_argument("--clock-step", type=int, default=100, help="SM clock step (MHz)")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for nvidia-smi / NVML")
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=15.0,
        help="Minimum wall-clock duration of the timed benchmark region at the "
        "slowest (--clock-min) clock; the iteration count is calibrated to hit this.",
    )
    parser.add_argument(
        "--calib-iterations",
        type=int,
        default=20,
        help="Iterations used for the one-time calibration run.",
    )
    parser.add_argument(
        "--calib-warmup",
        type=int,
        default=10,
        help="Warmup iterations used for the one-time calibration run.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=100,
        help="Warmup iterations for each per-clock benchmark run.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Sleep after locking a new clock before sampling/benchmarking.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=20.0,
        help="NVML sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="clock_sweep_results.csv",
        help="Path to write the CSV results table.",
    )
    parser.add_argument(
        "--no-reset-clock",
        action="store_true",
        help="Don't reset the GPU clock (nvidia-smi -rgc) at the end of the sweep.",
    )
    return parser


def clock_range(clock_min: int, clock_max: int, clock_step: int):
    clocks = list(range(clock_min, clock_max + 1, clock_step))
    if clocks[-1] != clock_max:
        clocks.append(clock_max)
    return clocks


def flops_for(mnkl):
    m, n, k, l = mnkl
    return 2 * m * n * k * l


def run_benchmark(ctx, iterations, warmup_iterations):
    workspace_count = 1
    if ctx.use_cold_l2:
        workspace_count = testing.get_workspace_count(
            ctx.one_workspace_bytes, warmup_iterations, iterations
        )
    exec_time_us = testing.benchmark(
        ctx.compiled_fn,
        workspace_generator=ctx.generate_tensors,
        workspace_count=workspace_count,
        stream=ctx.current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time_us


def _invoke_dense_gemm(kernel_module, kernel_args, sweep_args, sweep_callback):
    kernel_module.run(
        kernel_args.mnkl,
        kernel_args.a_dtype,
        kernel_args.b_dtype,
        kernel_args.c_dtype,
        kernel_args.acc_dtype,
        kernel_args.a_major,
        kernel_args.b_major,
        kernel_args.c_major,
        kernel_args.mma_tiler,
        kernel_args.mma_inst_shape,
        kernel_args.cluster_shape_mn,
        kernel_args.swizzle_size,
        kernel_args.raster_order,
        kernel_args.use_2cta_instrs,
        kernel_args.use_tma_store,
        kernel_args.tolerance,
        sweep_args.warmup_iterations,
        sweep_args.calib_iterations,
        kernel_args.skip_ref_check,
        kernel_args.use_cold_l2,
        kernel_args.benchmark == "default",
        kernel_args.init_normal,
        kernel_args.normal_mean,
        kernel_args.normal_std,
        benchmark_sweep=sweep_callback,
    )


def _invoke_blockscaled_gemm(kernel_module, kernel_args, sweep_args, sweep_callback):
    kernel_module.run(
        kernel_args.mnkl,
        kernel_args.a_dtype,
        kernel_args.b_dtype,
        kernel_args.sf_dtype,
        kernel_args.sf_vec_size,
        kernel_args.c_dtype,
        kernel_args.a_major,
        kernel_args.b_major,
        kernel_args.c_major,
        kernel_args.mma_tiler,
        kernel_args.mma_inst_shape,
        kernel_args.cluster_shape_mn,
        kernel_args.swizzle_size,
        kernel_args.raster_order,
        kernel_args.scheduler,
        kernel_args.tolerance,
        sweep_args.warmup_iterations,
        sweep_args.calib_iterations,
        kernel_args.skip_ref_check,
        kernel_args.use_cold_l2,
        kernel_args.init_normal,
        kernel_args.normal_mean,
        kernel_args.normal_std,
        kernel_args.prefetch_dist,
        benchmark_sweep=sweep_callback,
    )


# Each supported kernel module's run() has a different (positional) signature, so a
# small per-module adapter maps parsed CLI args -> a run(..., benchmark_sweep=...) call.
# To sweep a new kernel: give it a prepare_parser()/run(..., benchmark_sweep=...) hook
# (see BenchmarkContext in dense_gemm/dense_gemm_persistent.py) and register an adapter here.
KERNEL_ADAPTERS = {
    "dense_gemm.dense_gemm_persistent": _invoke_dense_gemm,
    "blockscaled_gemm.dense_blockscaled_gemm_persistent": _invoke_blockscaled_gemm,
}


def main():
    sweep_parser = build_sweep_parser()
    sweep_args, kernel_argv = sweep_parser.parse_known_args()
    if kernel_argv and kernel_argv[0] == "--":
        kernel_argv = kernel_argv[1:]

    kernel_module = importlib.import_module(sweep_args.module)
    invoke_run = KERNEL_ADAPTERS.get(sweep_args.module)
    if invoke_run is None:
        raise SystemExit(
            f"No sweep adapter registered for --module={sweep_args.module!r}. "
            f"Known modules: {sorted(KERNEL_ADAPTERS)}. Add an adapter to "
            f"KERNEL_ADAPTERS in {__file__}."
        )
    prepare_parser_fn = getattr(kernel_module, "prepare_parser", None) or getattr(
        kernel_module, "prepare_parser"
    )
    kernel_parser = prepare_parser_fn()
    kernel_args = kernel_parser.parse_args(kernel_argv)

    if len(kernel_args.mnkl) != 4:
        kernel_parser.error("--mnkl must contain exactly 4 values")

    total_flops = flops_for(kernel_args.mnkl)
    clocks = clock_range(sweep_args.clock_min, sweep_args.clock_max, sweep_args.clock_step)
    sampler = NvmlSampler(
        gpu_index=sweep_args.gpu_index, interval_s=sweep_args.sample_interval_ms / 1000.0
    )

    results = []

    def sweep_callback(ctx):
        # One-time calibration at the *unlocked* (default/boost) clock, i.e. the
        # fastest the kernel will ever run. Sizing the iteration count off the
        # slowest requested clock would undershoot --target-seconds at every
        # faster clock (fewer iterations x less time-per-iteration = short runs);
        # sizing off the fastest case guarantees every locked clock in the sweep,
        # being <= the unlocked clock, runs for at least --target-seconds.
        print(
            f"[sweep] Calibrating at unlocked/default clock "
            f"({sweep_args.calib_warmup} warmup + {sweep_args.calib_iterations} iterations)..."
        )
        reset_sm_clock(sweep_args.gpu_index)
        time.sleep(sweep_args.settle_seconds)
        calib_exec_time_us = run_benchmark(
            ctx, sweep_args.calib_iterations, sweep_args.calib_warmup
        )
        iterations = max(
            sweep_args.calib_iterations,
            int(-(-sweep_args.target_seconds * 1e6 // calib_exec_time_us)),  # ceil div
        )
        print(
            f"[sweep] Calibration: {calib_exec_time_us:.2f} us/iter (unlocked) -> using "
            f"{iterations} iterations per clock ({sweep_args.warmup_iterations} warmup)."
        )

        for clock_mhz in clocks:
            print(f"[sweep] Locking SM clock to {clock_mhz} MHz...")
            lock_sm_clock(clock_mhz, sweep_args.gpu_index)
            time.sleep(sweep_args.settle_seconds)

            sampler.start()
            exec_time_us = run_benchmark(
                ctx, iterations, sweep_args.warmup_iterations
            )
            stats = sampler.stop()

            tflops = total_flops / exec_time_us / 1e6
            row = {
                "requested_clock_mhz": clock_mhz,
                "achieved_mean_sm_clock_mhz": round(stats.mean_sm_clock_mhz, 1),
                "achieved_min_sm_clock_mhz": stats.min_sm_clock_mhz,
                "achieved_max_sm_clock_mhz": stats.max_sm_clock_mhz,
                "mean_power_w": round(stats.mean_power_w, 1),
                "max_power_w": round(stats.max_power_w, 1),
                "mean_temperature_c": round(stats.mean_temperature_c, 1),
                "exec_time_us_per_iter": round(exec_time_us, 2),
                "tflops": round(tflops, 1),
                "iterations": iterations,
                "num_nvml_samples": stats.num_samples,
                "sampled_duration_s": round(stats.duration_s, 2),
            }
            results.append(row)
            print(
                f"[sweep] clock={clock_mhz} MHz  "
                f"achieved={row['achieved_mean_sm_clock_mhz']} MHz  "
                f"power={row['mean_power_w']} W  "
                f"{row['tflops']} TFLOP/s  "
                f"({row['num_nvml_samples']} samples over {row['sampled_duration_s']}s)"
            )

        return results

    try:
        invoke_run(kernel_module, kernel_args, sweep_args, sweep_callback)
    finally:
        if not sweep_args.no_reset_clock:
            print("[sweep] Resetting GPU clock...")
            reset_sm_clock(sweep_args.gpu_index)

    if not results:
        print("[sweep] No results collected.")
        return

    fieldnames = list(results[0].keys())
    with open(sweep_args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[sweep] Wrote {sweep_args.csv_out}")

    header = (
        f"{'req MHz':>8} {'avg MHz':>9} {'min MHz':>9} {'max MHz':>9} "
        f"{'avg W':>8} {'max W':>8} {'avg C':>7} {'us/iter':>10} {'TFLOP/s':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['requested_clock_mhz']:>8} "
            f"{row['achieved_mean_sm_clock_mhz']:>9} "
            f"{row['achieved_min_sm_clock_mhz']:>9} "
            f"{row['achieved_max_sm_clock_mhz']:>9} "
            f"{row['mean_power_w']:>8} "
            f"{row['max_power_w']:>8} "
            f"{row['mean_temperature_c']:>7} "
            f"{row['exec_time_us_per_iter']:>10} "
            f"{row['tflops']:>9}"
        )


if __name__ == "__main__":
    main()
