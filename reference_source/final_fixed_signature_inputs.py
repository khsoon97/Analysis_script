#!/usr/bin/env python3
"""Public fixed-signature input API.

Only explicitly pinned, read-only Step02/Step04B directional inputs are exposed
to the canonical scRNA signature backend. Latest-run discovery, Figure-selected
support, positive-only legacy scoring, and standalone weight-export interfaces
are intentionally absent from this public module.
"""

from final_fixed_signature_input_core import (
    COVERAGE_PATHS,
    STEP02_ALE_ROOT,
    STEP04B_ALE_ROOT,
    build_step02_broad_directional_signature_support,
    build_step04b_tcell_directional_signature_support,
    collapse_directional_signature_support_to_proxy_groups,
    load_ale_directional_step02_broad_class_weights,
    load_ale_directional_step04b_class_weights,
)

__all__ = [
    "COVERAGE_PATHS",
    "STEP02_ALE_ROOT",
    "STEP04B_ALE_ROOT",
    "build_step02_broad_directional_signature_support",
    "build_step04b_tcell_directional_signature_support",
    "collapse_directional_signature_support_to_proxy_groups",
    "load_ale_directional_step02_broad_class_weights",
    "load_ale_directional_step04b_class_weights",
]
