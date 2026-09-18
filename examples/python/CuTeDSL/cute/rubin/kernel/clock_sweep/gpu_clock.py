# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for locking / resetting GPU SM clocks via `nvidia-smi -lgc/-rgc`."""

import subprocess


def lock_sm_clock(clock_mhz: int, gpu_index: int = 0):
    subprocess.run(
        [
            "sudo",
            "nvidia-smi",
            "-lgc",
            f"{clock_mhz},{clock_mhz}",
            "-i",
            str(gpu_index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def reset_sm_clock(gpu_index: int = 0):
    subprocess.run(
        [
            "sudo",
            "nvidia-smi",
            "-rgc",
            "-i",
            str(gpu_index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
