"""
Timing chain and vector binding operations for OMNIRING.
"""
from typing import List, Union
import numpy as np
from .types import PhaseVector


def bind(a: PhaseVector, b: PhaseVector) -> PhaseVector:
    """
    Hadamard (element-wise) complex product: A ⊙ B.
    Preserves unit modulus on the complex circle.
    """
    return a * b


def unbind(a: PhaseVector, b: PhaseVector) -> PhaseVector:
    """
    Unbinds vector B from A: A ⊙ B*.
    """
    return a * np.conj(b)


def encode(x: int, rotations: List[np.ndarray]) -> PhaseVector:
    """
    Encode an integer state x into a composite phase vector across all provided rings.
    enc(x) = exp(1j * sum_m(x * r_m))
    """
    total_angle = sum(x * r for r in rotations)
    return np.exp(1j * total_angle)


def get_tick(rotations: List[np.ndarray]) -> PhaseVector:
    """
    Construct the tick operator: tick = enc(1).
    """
    return encode(1, rotations)


def advance(v: PhaseVector, tick: PhaseVector, steps: int = 1) -> PhaseVector:
    """
    Advance a state vector by `steps` ticks.
    """
    if steps == 0:
        return v
    step_tick = tick ** steps
    return bind(v, step_tick)
