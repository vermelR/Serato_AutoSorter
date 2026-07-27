from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from serato_ai.settings import load_settings


pytestmark = pytest.mark.unit


def test_package_import_smoke_has_no_circular_imports() -> None:
    modules = [
        "serato_ai.core.models",
        "serato_ai.core.crate_filters",
        "serato_ai.core.confidence",
        "serato_ai.core.path_utils",
        "serato_ai.core.assignment_utils",
        "serato_ai.core.validation",
        "serato_ai.core.result_summary",
        "serato_ai.core.dataframes",
        "serato_ai.core.metadata_rules",
        "serato_ai.core.quality_rules",
        "serato_ai.core.training_rules",
        "serato_ai.infrastructure.serato_reader",
        "serato_ai.infrastructure.serato_writer",
        "serato_ai.infrastructure.tag_writer",
        "serato_ai.infrastructure.model_store",
        "serato_ai.infrastructure.application_data",
        "serato_ai.infrastructure.feature_cache_store",
        "serato_ai.infrastructure.feedback_store",
        "serato_ai.infrastructure.onboarding_store",
        "serato_ai.infrastructure.personal_model_store",
        "serato_ai.infrastructure.metadata_providers",
        "serato_ai.infrastructure.model_quality_store",
        "serato_ai.services.model_service",
        "serato_ai.services.prediction_service",
        "serato_ai.services.crate_assignment_service",
        "serato_ai.services.watcher_service",
        "serato_ai.services.library_scan_service",
        "serato_ai.services.feature_extraction_service",
        "serato_ai.services.training_dataset_service",
        "serato_ai.services.training_service",
        "serato_ai.services.model_evaluation_service",
        "serato_ai.services.model_activation_service",
        "serato_ai.services.onboarding_service",
        "serato_ai.services.feedback_service",
        "serato_ai.services.model_health_service",
        "serato_ai.services.metadata_enrichment_service",
        "serato_ai.services.model_quality_service",
        "serato_ai.ui.session_state",
    ]
    for name in modules:
        assert importlib.import_module(name)


def test_thin_app_entrypoint_has_no_low_level_writer_or_business_rules() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert "serato_writer" not in source
    assert "write_tracks_to_crates" not in source
    assert "parse_serato_crate" not in source
    assert "render_application" in source
    assert len(source.splitlines()) < 25


def test_settings_are_typed_and_do_not_embed_a_user_specific_path(tmp_path: Path) -> None:
    settings = load_settings({"SERATO_CRATE_MODEL": "custom.pkl"}, home=tmp_path / "user-home")
    assert settings.crate_model_path == "custom.pkl"
    assert settings.default_serato_root == str(tmp_path / "user-home" / "Music" / "_Serato_")
    assert "/Users/diora" not in inspect.getsource(load_settings)


def test_legacy_runtime_python_modules_have_no_hardcoded_diora_path() -> None:
    for path in Path(".").glob("*.py"):
        assert "/Users/diora" not in path.read_text(encoding="utf-8")
