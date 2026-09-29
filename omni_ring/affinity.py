"""
CPU Core Affinity and Thread Pacing for OMNIRING.
Prevents 15W thermal throttling on mobile/laptop CPUs (e.g. i5-8365U)
by pinning SIMD execution to physical cores and managing OpenMP threads.
"""
import os
import contextlib
from typing import Set, Sequence, Optional
from .avx2 import set_num_threads


def get_current_affinity() -> Set[int]:
    """Get current process CPU affinity mask."""
    if hasattr(os, "sched_getaffinity"):
        return os.sched_getaffinity(0)
    return set()


def set_physical_affinity(cores: Sequence[int] = (0, 1, 2, 3)) -> bool:
    """
    Pin the current process to specified physical cores.
    Returns True if successfully set, False if not supported on platform.
    """
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cores))
            return True
        except (OSError, PermissionError):
            return False
    return False


def configure_pacing(n_physical_cores: int = 4, pin: bool = True) -> dict:
    """
    Enforce thermal and compute pacing:
    1. Pins to the first N physical cores if pin=True.
    2. Limits OpenMP thread count to N.
    3. Sets OMP_WAIT_POLICY=PASSIVE to prevent thread spin-wait lockups.
    """
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    pinned = False
    if pin:
        cores = tuple(range(n_physical_cores))
        pinned = set_physical_affinity(cores)
    
    set_num_threads(n_physical_cores)
    
    return {
        "n_threads": n_physical_cores,
        "pinned": pinned,
        "active_affinity": list(get_current_affinity()) if hasattr(os, "sched_getaffinity") else []
    }


@contextlib.contextmanager
def pinned_pacing(n_physical_cores: int = 4):
    """Context manager to temporarily run with pinned physical cores and paced threads."""
    prev_affinity = get_current_affinity()
    try:
        configure_pacing(n_physical_cores=n_physical_cores, pin=True)
        yield
    finally:
        if prev_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, prev_affinity)
            except (OSError, PermissionError):
                pass
