"""
utils/system_monitor.py
Collects CPU, RAM, and GPU (RTX 3060) metrics via psutil and pynvml.

Uses a background thread that continuously samples metrics every 250 ms so
that any caller gets an instantly-fresh reading with zero blocking time.
"""
import logging
import threading
import time
from dataclasses import dataclass

import psutil

logger = logging.getLogger("TradingSystem.SystemMonitor")

# Suppress the pynvml→nvidia-ml-py rename FutureWarning (cosmetic only)
import warnings
warnings.filterwarnings(
    "ignore", message="The pynvml package is deprecated.*", category=FutureWarning
)

_gpu_available = False
_last_gpu_error_log = 0.0
_last_sampler_error_log = 0.0
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _gpu_available = True
    gpu_name = pynvml.nvmlDeviceGetName(_GPU_HANDLE)
    logger.info(f"GPU monitor initialised: {gpu_name}")
except Exception as e:
    logger.warning(f"GPU monitoring unavailable: {e}")
    _GPU_HANDLE = None


@dataclass
class SystemMetrics:
    cpu_pct: float          # 0–100
    ram_used_gb: float
    ram_total_gb: float
    ram_pct: float          # 0–100
    gpu_util_pct: float     # 0–100  (0 if no GPU)
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    gpu_mem_pct: float      # 0–100
    gpu_temp_c: float       # degrees Celsius


# ── Background sampler ────────────────────────────────────────────
# Continuously pre-samples metrics every SAMPLE_INTERVAL_S seconds
# so `get_system_metrics()` is a pure zero-latency cache read.

SAMPLE_INTERVAL_S = 0.25   # 250 ms — matches Task Manager refresh rate

_lock = threading.Lock()
_latest: SystemMetrics = SystemMetrics(
    cpu_pct=0.0, ram_used_gb=0.0, ram_total_gb=0.0, ram_pct=0.0,
    gpu_util_pct=0.0, gpu_mem_used_mb=0.0, gpu_mem_total_mb=0.0,
    gpu_mem_pct=0.0, gpu_temp_c=0.0,
)

# Kick-start psutil CPU tracking so first reading is meaningful
psutil.cpu_percent(interval=None)


def _sample_once() -> SystemMetrics:
    """Performs a single blocking sample of all hardware metrics."""
    global _last_gpu_error_log
    # CPU — uses blocking 0.1 s measurement for accurate per-sample reading
    cpu = psutil.cpu_percent(interval=0.1)

    # RAM
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)

    # GPU
    gpu_util = gpu_mem_used = gpu_mem_total = gpu_temp = 0.0
    if _gpu_available and _GPU_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            temp = pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            gpu_util      = float(util.gpu)
            gpu_mem_used  = mem.used  / (1024 ** 2)
            gpu_mem_total = mem.total / (1024 ** 2)
            gpu_temp      = float(temp)
        except Exception as exc:
            now_mono = time.monotonic()
            if now_mono - _last_gpu_error_log >= 60.0:
                logger.warning("GPU metric sampling failed: %s", exc)
                _last_gpu_error_log = now_mono

    gpu_mem_pct = (gpu_mem_used / gpu_mem_total * 100) if gpu_mem_total > 0 else 0.0

    return SystemMetrics(
        cpu_pct=cpu,
        ram_used_gb=round(ram_used, 2),
        ram_total_gb=round(ram_total, 2),
        ram_pct=round(ram.percent, 1),
        gpu_util_pct=round(gpu_util, 1),
        gpu_mem_used_mb=round(gpu_mem_used, 1),
        gpu_mem_total_mb=round(gpu_mem_total, 1),
        gpu_mem_pct=round(gpu_mem_pct, 1),
        gpu_temp_c=round(gpu_temp, 1),
    )


def _background_sampler():
    """Daemon thread: continuously refreshes `_latest` every SAMPLE_INTERVAL_S."""
    global _latest, _last_sampler_error_log
    while True:
        try:
            m = _sample_once()
            with _lock:
                _latest = m
        except Exception as exc:
            now_mono = time.monotonic()
            if now_mono - _last_sampler_error_log >= 60.0:
                logger.warning("System metric sampler failed: %s", exc)
                _last_sampler_error_log = now_mono
        # _sample_once already spent ~0.1 s on CPU measurement;
        # sleep the remainder so the total cycle is SAMPLE_INTERVAL_S
        time.sleep(max(0.0, SAMPLE_INTERVAL_S - 0.1))


# Start the daemon thread once at import time
_sampler_thread = threading.Thread(target=_background_sampler, daemon=True, name="SystemMonitorSampler")
_sampler_thread.start()


def get_system_metrics() -> SystemMetrics:
    """Instant zero-latency read of the most recent hardware sample."""
    with _lock:
        return _latest
