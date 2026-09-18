# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable background NVML sampler for GPU clock / power measurements.

Usage:
    sampler = NvmlSampler(gpu_index=0, interval_s=0.02)
    sampler.start()
    ... run the region of interest ...
    stats = sampler.stop()
    print(stats.mean_sm_clock_mhz, stats.mean_power_w)
"""

import dataclasses
import threading
import time
from typing import List, Optional

import pynvml


@dataclasses.dataclass
class SweepSample:
    t: float
    sm_clock_mhz: int
    graphics_clock_mhz: int
    power_w: float
    temperature_c: int


@dataclasses.dataclass
class SweepStats:
    num_samples: int
    duration_s: float
    mean_sm_clock_mhz: float
    min_sm_clock_mhz: float
    max_sm_clock_mhz: float
    mean_power_w: float
    max_power_w: float
    mean_temperature_c: float
    samples: List[SweepSample]


class NvmlSampler:
    """Samples SM clock and power draw for one GPU at a fixed interval on a background thread."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.02):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[SweepSample] = []
        self._owns_nvml_init = False

    def _ensure_nvml_init(self):
        try:
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            pynvml.nvmlInit()
            self._owns_nvml_init = True

    def _sample_loop(self):
        handle = self._handle
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                gr_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    temp_c = -1
                self._samples.append(
                    SweepSample(
                        t=time.monotonic() - t0,
                        sm_clock_mhz=sm_clock,
                        graphics_clock_mhz=gr_clock,
                        power_w=power_w,
                        temperature_c=temp_c,
                    )
                )
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.interval_s)

    def start(self):
        self._ensure_nvml_init()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> SweepStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        samples = self._samples
        if not samples:
            return SweepStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        sm_clocks = [s.sm_clock_mhz for s in samples]
        powers = [s.power_w for s in samples]
        temps = [s.temperature_c for s in samples if s.temperature_c >= 0]
        return SweepStats(
            num_samples=len(samples),
            duration_s=samples[-1].t - samples[0].t,
            mean_sm_clock_mhz=sum(sm_clocks) / len(sm_clocks),
            min_sm_clock_mhz=min(sm_clocks),
            max_sm_clock_mhz=max(sm_clocks),
            mean_power_w=sum(powers) / len(powers),
            max_power_w=max(powers),
            mean_temperature_c=(sum(temps) / len(temps)) if temps else -1.0,
            samples=samples,
        )
