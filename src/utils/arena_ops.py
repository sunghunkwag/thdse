"""
Arena operations — pure-Python utilities for FHRR phase algebra.

These utilities operate on phase arrays (List[float]) alongside the Rust arena,
enabling operations the arena doesn't expose (conjugate, phase extraction).

No raw buffer access required. All operations derive from the algebraic
properties of FHRR (Fourier Holographic Reduced Representation):
  - bind(A, B) phases  = A_phases + B_phases  (element-wise addition)
  - bundle({Aᵢ}) phases = atan2(Σ sin(θᵢ), Σ cos(θᵢ))  per dimension
  - conjugate(A) phases = -A_phases  (negation)

In FHRR, every vector has unit magnitude per component: (cos θ, sin θ).
The conjugate (= inverse for unit-magnitude) is (cos(-θ), sin(-θ)).
bind(V, conjugate(V)) = identity (all phases ≈ 0).
"""

from typing import Any, List

import numpy as np


def bind_phases(phases_a: List[float], phases_b: List[float]) -> List[float]:
    """Compute the phase array of bind(A, B) = element-wise complex multiply.

    In FHRR: θ_result[d] = θ_A[d] + θ_B[d]
    """
    return [a + b for a, b in zip(phases_a, phases_b)]


def bundle_phases(phase_arrays: List[List[float]]) -> List[float]:
    """Compute the phase array of bundle({V₁, ..., Vₙ}) = sum + normalize.

    In FHRR: θ_result[d] = atan2(Σᵢ sin(θᵢ[d]), Σᵢ cos(θᵢ[d]))

    This is exact (not approximate) because the arena's normalize step
    projects each component back to unit magnitude, which is equivalent
    to taking the angle of the complex sum.
    """
    if not phase_arrays:
        return []
    dim = len(phase_arrays[0])
    cos_sum = np.zeros(dim)
    sin_sum = np.zeros(dim)
    for phases in phase_arrays:
        arr = np.asarray(phases)
        cos_sum += np.cos(arr)
        sin_sum += np.sin(arr)
    return np.arctan2(sin_sum, cos_sum).tolist()


def negate_phases(phases: List[float]) -> List[float]:
    """Compute the conjugate phases: -θ per dimension.

    bind(V, conjugate(V)) produces the identity vector (all phases ≈ 0).
    """
    return [-p for p in phases]


def conjugate_into(arena: Any, phases: List[float], dst_handle: int):
    """Inject the complex conjugate of a phase vector into an arena slot.

    The conjugate of FHRR vector with phases [θ₁, θ₂, ...] is [-θ₁, -θ₂, ...].
    This is the multiplicative inverse for unit-magnitude FHRR vectors:
        bind(V, conjugate(V)) ≈ identity (phases ≈ 0, i.e., all-real ≈ 1+0j)
    """
    arena.inject_phases(dst_handle, negate_phases(phases))
