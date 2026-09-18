#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

SWEEP_ARGS="--gpu-index 0 --clock-min 1200 --clock-max 2400 --clock-step 100 --warmup-seconds 10 --bench-iterations 100"
COMMON_ARGS="--use_cold_l2 --init_normal --normal_mean=0.0 --normal_std=0.01 --a_major=k --b_major=k --c_major=m --cluster_shape_mn=2,1 --swizzle_size=1 --raster_order=m --skip_ref_check"

set -x

python3 clock_sweep/sweep_clocks.py --module dense_gemm.dense_gemm_persistent $SWEEP_ARGS --csv-out bf16_dense_gemm_clock_sweep.csv -- $COMMON_ARGS --mnkl=4096,13568,16384,1 --a_dtype=BFloat16 --b_dtype=BFloat16 --c_dtype=BFloat16 --acc_dtype=Float32 --use_tma_store --use_2cta_instrs --mma_tiler=256,256,64 --mma_inst_shape=256,256,16

python3 clock_sweep/sweep_clocks.py --module dense_gemm.dense_gemm_persistent $SWEEP_ARGS --csv-out fp8_dense_gemm_clock_sweep.csv -- $COMMON_ARGS --mnkl=4096,13568,32768,1 --a_dtype=Float8E4M3FN --b_dtype=Float8E4M3FN --c_dtype=BFloat16 --acc_dtype=Float32 --use_tma_store --use_2cta_instrs --mma_tiler=512,256,128 --mma_inst_shape=256,256,64

python3 clock_sweep/sweep_clocks.py --module blockscaled_gemm.dense_blockscaled_gemm_persistent $SWEEP_ARGS --csv-out mxfp8_dense_gemm_clock_sweep.csv -- $COMMON_ARGS --mnkl=4096,13568,32768,1 --a_dtype=Float8E4M3FN --b_dtype=Float8E4M3FN --sf_dtype=Float8E8M0FNU --sf_vec_size=32 --c_dtype=BFloat16 --mma_tiler=512,256,128 --mma_inst_shape=256,256,64 --prefetch_dist=0

python3 clock_sweep/sweep_clocks.py --module blockscaled_gemm.dense_blockscaled_gemm_persistent $SWEEP_ARGS --csv-out nvfp4_dense_gemm_clock_sweep.csv -- $COMMON_ARGS --mnkl=4096,13568,32768,1 --a_dtype=Float4E2M1FN --b_dtype=Float4E2M1FN --sf_dtype=Float8E4M3FN --sf_vec_size=16 --c_dtype=BFloat16 --mma_tiler=512,256,256 --mma_inst_shape=256,256,128 --prefetch_dist=0