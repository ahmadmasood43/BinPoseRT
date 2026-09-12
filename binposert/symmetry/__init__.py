"""SymmetryGroup construction and symmetry-aware pose alignment (ADR-0003)."""

from binposert.symmetry.group import (
    align_to_reference,
    from_bop_model_info,
    sym_aware_rotation_distance_deg,
    sym_aware_se3_distance,
)

__all__ = [
    "align_to_reference",
    "from_bop_model_info",
    "sym_aware_rotation_distance_deg",
    "sym_aware_se3_distance",
]
