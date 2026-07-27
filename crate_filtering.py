"""Backward-compatible imports for pure crate-filtering rules.

New code should import :mod:`serato_ai.core.crate_filters` directly.
"""

from serato_ai.core.crate_filters import (
    allowed_crates_for_categories,
    apply_explicit_exclusions,
    crate_category,
    crate_filter_signature,
    filter_crate_selections,
    format_crate_label,
    normalize_crate_names,
)

__all__ = [
    "allowed_crates_for_categories", "apply_explicit_exclusions", "crate_category",
    "crate_filter_signature", "filter_crate_selections", "format_crate_label",
    "normalize_crate_names",
]
