"""
Redundant Check Ring Anomaly and Slip Detection for OMNIRING.
"""
from typing import List, Tuple
from .types import RingSpec, DEFAULT_SPEC
from .crt import solve_crt


class ResidueDesyncError(Exception):
    """Raised when an inconsistent or corrupted residue state is detected."""
    pass


def check_anomaly(residues: List[int], spec: RingSpec = DEFAULT_SPEC) -> int:
    """
    Verify the consistency of a 5-residue tuple [r3, r5, r7, r13, r17].
    
    In the valid state space, any true value x in [0, N_info) produces
    consistent residues across all rings, including the check ring m_check = 17.
    Since M_total = 23,205 and N_info = 1,365, any single-ring error will project
    the reconstructed CRT value into the range [N_info, M_total - 1], triggering
    a ResidueDesyncError.

    Returns:
        The verified coordinate X in [0, N_info - 1].
    Raises:
        ResidueDesyncError: If X >= N_info.
    """
    all_mods = spec.all_moduli
    if len(residues) != len(all_mods):
        raise ValueError(f"Expected {len(all_mods)} residues, got {len(residues)}")

    X = solve_crt(residues, all_mods)
    if X >= spec.data_capacity:
        raise ResidueDesyncError(
            f"Phase slip / corrupt residue detected: CRT resolved to {X}, "
            f"exceeding valid information capacity {spec.data_capacity}"
        )
    return X


def run_full_error_detection_test(spec: RingSpec = DEFAULT_SPEC) -> Tuple[int, int]:
    """
    Exhaustively test all possible single-ring errors across all valid states.
    For N_info = 1365, total corruptions = 1365 * ( (3-1) + (5-1) + (7-1) + (13-1) + (17-1) ) = 54,600.
    
    Returns:
        (missed_errors, total_errors)
    """
    all_mods = spec.all_moduli
    n_info = spec.data_capacity
    missed = 0
    total = 0

    for x in range(n_info):
        base_res = [x % m for m in all_mods]
        for i, m in enumerate(all_mods):
            for err in range(1, m):
                corrupt_res = list(base_res)
                corrupt_res[i] = (corrupt_res[i] + err) % m
                total += 1
                try:
                    check_anomaly(corrupt_res, spec)
                    # If no exception, it's a missed error
                    missed += 1
                except ResidueDesyncError:
                    pass

    return missed, total
