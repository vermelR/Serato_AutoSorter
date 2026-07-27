from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from crate_filtering import (
    allowed_crates_for_categories,
    apply_explicit_exclusions,
    filter_crate_selections,
)
from phase4_engine import propose_crates_for_files
from watcher import Watcher


pytestmark = [pytest.mark.unit, pytest.mark.prediction, pytest.mark.watcher]


class FakeModel:
    classes_ = ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]

    def predict_proba(self, _features):
        return [[0.5, 0.3, 0.2]]


@pytest.fixture
def bundle() -> dict:
    return {"model": FakeModel(), "feature_columns": ["feature"]}


@pytest.fixture
def fake_prediction_dependencies():
    with (
        patch("phase4_engine.read_track_metadata", return_value={
            "title": "Test", "artist": "Artist", "genre": "", "year": "",
        }),
        patch("phase4_engine.extract_audio_features", return_value=[123.0]),
        patch("phase4_engine.identify_genre_year", return_value={
            "genre": "", "year": "", "genre_source": "tag", "year_source": "tag",
        }),
    ):
        yield


def test_categories_create_an_allow_list_then_apply_explicit_exclusions() -> None:
    options = ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]
    category_allowed = allowed_crates_for_categories(options, ["House"])

    assert category_allowed == ["House%%Club", "House%%Deep"]
    assert apply_explicit_exclusions(category_allowed, {"House%%Deep"}) == ["House%%Club"]
    assert filter_crate_selections(
        ["House%%Club", "Hip Hop%%Open Format", "House%%Club"], category_allowed
    ) == ["House%%Club"]


def test_empty_category_selection_allows_every_model_crate() -> None:
    options = ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]
    assert allowed_crates_for_categories(options, []) == options


def test_categories_use_exact_top_level_matches_and_support_multiple_values() -> None:
    options = ["House%%Club", "Deep House%%Late", "Hip Hop%%Open Format", "Techno%%Peak"]
    assert allowed_crates_for_categories(options, ["House"]) == ["House%%Club"]
    assert allowed_crates_for_categories(options, ["House", "Techno"]) == [
        "House%%Club", "Techno%%Peak",
    ]


def test_model_predictions_only_include_allowed_categories_and_renormalize_confidence(
    bundle: dict, fake_prediction_dependencies,
) -> None:
    rows, failures = propose_crates_for_files(
        bundle,
        [Path("/tmp/track.mp3")],
        topk=3,
        identify_genre=True,
        allowed_crates={"House%%Club", "House%%Deep"},
    )

    assert failures.empty
    row = rows.iloc[0]
    assert row["Suggested Crate"] == "House%%Club"
    assert row["_top1_crate"] == "House%%Club"
    assert row["_top2_crate"] == "House%%Deep"
    assert "Hip Hop%%Open Format" not in row["_allowed_crates"]
    assert row["Confidence"] == pytest.approx(0.5 / 0.7)
    assert row["_top1_prob"] == pytest.approx(0.5 / 0.7)
    assert row["_top2_prob"] == pytest.approx(0.2 / 0.7)


def test_explicit_exclusion_applies_after_category_allow_list(
    bundle: dict, fake_prediction_dependencies,
) -> None:
    rows, failures = propose_crates_for_files(
        bundle,
        [Path("/tmp/track.mp3")],
        topk=3,
        identify_genre=True,
        allowed_crates={"House%%Club", "House%%Deep"},
        excluded_crates={"House%%Club"},
    )

    assert failures.empty
    row = rows.iloc[0]
    assert row["Suggested Crate"] == "House%%Deep"
    assert row["Confidence"] == pytest.approx(1.0)
    assert row["_allowed_crates"] == ["House%%Deep"]


def test_watcher_passes_current_allow_list_and_exclusions_to_predictions() -> None:
    watcher = Watcher(
        folders=[],
        allowed_crates={"House%%Club"},
        excluded_crates={"House%%Deep"},
    )
    watcher._bundle = {"model": object()}
    watcher._candidate_files = lambda: [Path("/tmp/watched.mp3")]  # type: ignore[method-assign]
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]

    with (
        patch("watcher.propose_crates_for_files", return_value=(pd.DataFrame(), pd.DataFrame())) as propose,
        patch("watcher.append_pending"),
    ):
        watcher._scan_once()

    assert propose.call_args.kwargs["allowed_crates"] == {"House%%Club"}
    assert propose.call_args.kwargs["excluded_crates"] == {"House%%Deep"}

    watcher.set_crate_filters(
        allowed_crates={"Hip Hop%%Open Format"}, excluded_crates=set()
    )
    assert watcher.allowed_crates == {"Hip Hop%%Open Format"}
    assert watcher.excluded_crates == set()


def test_watcher_keeps_candidates_pending_when_no_crates_are_allowed() -> None:
    watcher = Watcher(folders=[], allowed_crates=set())
    watcher._bundle = {"model": object()}
    watcher._candidate_files = lambda: [Path("/tmp/watched.mp3")]  # type: ignore[method-assign]

    with patch("watcher.propose_crates_for_files") as propose:
        watcher._scan_once()

    propose.assert_not_called()
