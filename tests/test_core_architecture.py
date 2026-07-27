from __future__ import annotations

import importlib
import inspect

import pytest

from serato_ai.core.assignment_utils import expand_assignments, unique_tracks
from serato_ai.core.confidence import filter_and_rank_probabilities, format_confidence
from serato_ai.core.crate_filters import format_crate_label
from serato_ai.core.dataframes import SHOW_COLS, csv_export_frame, predictions_to_dataframe, prepare_prediction_frame
from serato_ai.core.models import ApprovedTrack, CrateSuggestion, TrackPrediction
from serato_ai.core.validation import validate_approved_tracks


pytestmark = pytest.mark.unit


def test_core_modules_import_without_streamlit_or_filesystem_side_effects() -> None:
    for module in (
        "serato_ai.core.models",
        "serato_ai.core.crate_filters",
        "serato_ai.core.confidence",
        "serato_ai.core.path_utils",
        "serato_ai.core.assignment_utils",
        "serato_ai.core.validation",
        "serato_ai.core.result_summary",
    ):
        imported = importlib.import_module(module)
        assert "streamlit" not in inspect.getsource(imported).lower()


def test_core_probability_filtering_and_confidence_use_raw_crate_names() -> None:
    suggestions = filter_and_rank_probabilities(
        [("House%%Club", 0.5), ("Hip Hop%%Open", 0.3), ("House%%Deep", 0.2)],
        allowed_crates=["House%%Club", "House%%Deep"],
    )
    assert [(item.crate_name, item.probability) for item in suggestions] == [
        ("House%%Club", pytest.approx(0.5 / 0.7)),
        ("House%%Deep", pytest.approx(0.2 / 0.7)),
    ]
    assert format_crate_label(suggestions[0].crate_name) == "House › Club"
    assert format_confidence(suggestions[0].probability) == "71.4%"


def test_core_assignment_expansion_deduplicates_without_using_suggestions() -> None:
    tracks = [
        ApprovedTrack("/tmp/one.mp3", ("House%%Club", "House%%Deep", "House%%Club"), "House", "2024"),
        ApprovedTrack("/tmp/one.mp3", ("House%%Deep",), "Other", "1999"),
    ]
    assert [(item.crate_name, item.track_path) for item in expand_assignments(tracks)] == [
        ("House%%Club", "/tmp/one.mp3"),
        ("House%%Deep", "/tmp/one.mp3"),
    ]
    assert unique_tracks(tracks) == [tracks[0]]


def test_core_validation_removes_disallowed_crates_and_reports_missing_selection() -> None:
    valid, issues = validate_approved_tracks(
        [
            ApprovedTrack("/tmp/allowed.mp3", ("House%%Club", "Hip Hop%%Open")),
            ApprovedTrack("/tmp/blocked.mp3", ("Hip Hop%%Open",)),
        ],
        ["House%%Club"],
    )
    assert valid == [ApprovedTrack("/tmp/allowed.mp3", ("House%%Club",))]
    assert issues[0].code == "missing_final_crates"
    assert issues[0].track_path == "/tmp/blocked.mp3"


def test_typed_prediction_dataframe_conversion_preserves_review_and_csv_contracts() -> None:
    prediction = TrackPrediction(
        path="/tmp/song.mp3", suggested_crate="House%%Club", confidence=0.825,
        suggestions=(CrateSuggestion("House%%Club", 0.825, 1),), title="Song", artist="Artist",
        allowed_crates=("House%%Club",),
    )
    selections: dict[str, list[str]] = {}
    frame = prepare_prediction_frame(
        predictions_to_dataframe([prediction]), selections, 5, ["House%%Club"],
        crate_options=["House%%Club"],
    )
    assert set(SHOW_COLS).issubset(frame.columns)
    assert frame.iloc[0]["Top Suggested Crate"] == "House › Club"
    assert frame.iloc[0]["Final Crates"] == ["House%%Club"]
    exported = csv_export_frame(frame)
    assert exported.iloc[0]["Final Crates"] == "House › Club"
    assert not any(column.startswith("_top") for column in exported.columns)
