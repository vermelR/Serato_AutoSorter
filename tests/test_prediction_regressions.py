from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from phase4_engine import (
    collect_audio_files,
    identify_genre_year,
    load_model_bundle,
    normalize_path,
    propose_crates_for_files,
)


pytestmark = [pytest.mark.unit, pytest.mark.prediction]


class FixedModel:
    classes_ = np.array(["House%%Club", "Hip Hop%%Open Format", "House%%Deep"])

    def __init__(self, probabilities=(0.5, 0.3, 0.2)):
        self.probabilities = probabilities

    def predict_proba(self, _features):
        return np.array([self.probabilities])


class FailingModel:
    classes_ = np.array(["House%%Club"])

    def predict_proba(self, _features):
        raise RuntimeError("predict_proba exploded")


@pytest.fixture
def prediction_dependencies():
    with (
        patch("phase4_engine.read_track_metadata", return_value={
            "title": "Test", "artist": "Artist", "genre": "Tagged", "year": "2020",
        }),
        patch("phase4_engine.extract_audio_features", return_value=[123.0]),
    ):
        yield


def test_collect_audio_files_handles_recursive_nonrecursive_and_unsupported_files(
    temporary_music_library: Path,
) -> None:
    assert [path.name for path in collect_audio_files([str(temporary_music_library)], recursive=False)] == [
        "Artist - One.mp3"
    ]
    assert {path.name for path in collect_audio_files([str(temporary_music_library)], recursive=True)} == {
        "Artist - One.mp3", "Artist - Two.flac",
    }
    assert collect_audio_files([str(temporary_music_library / "ignore.txt")]) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Users/example/song.mp3", "/Users/example/song.mp3"),
        ("file:///tmp/A%20Song.mp3", "/tmp/A Song.mp3"),
        ("Macintosh HD:Users:example:song.mp3", "/Users/example/song.mp3"),
    ],
)
def test_prediction_path_normalization_handles_serato_variants(raw: str, expected: str) -> None:
    assert normalize_path(raw) == expected


def test_valid_temporary_model_loads(temporary_model_file: Path) -> None:
    bundle = load_model_bundle(str(temporary_model_file))
    assert list(bundle["model"].classes_) == ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]


@pytest.mark.parametrize("payload", [{}, {"model": object(), "feature_columns": ["feature"]}, {"model": FixedModel(), "feature_columns": "feature"}])
def test_model_bundle_validation_has_clear_errors(tmp_path: Path, payload: dict) -> None:
    model_path = tmp_path / "invalid.pkl"
    joblib.dump(payload, model_path)
    with pytest.raises(ValueError):
        load_model_bundle(str(model_path))


def test_missing_and_corrupt_model_fail_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_model_bundle(str(tmp_path / "missing.pkl"))
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a joblib pickle")
    with pytest.raises(Exception):
        load_model_bundle(str(corrupt))


def test_prediction_is_deterministic_filtered_renormalized_and_topk_capped(prediction_dependencies) -> None:
    bundle = {"model": FixedModel(), "feature_columns": ["feature"]}
    rows, failures = propose_crates_for_files(
        bundle,
        [Path("/tmp/track.mp3")],
        topk=99,
        identify_genre=False,
        allowed_crates={"House%%Club", "House%%Deep"},
    )
    assert failures.empty
    row = rows.iloc[0]
    assert row["Suggested Crate"] == "House%%Club"
    assert row["Confidence"] == pytest.approx(0.5 / 0.7)
    assert row["_top1_prob"] == row["Confidence"]
    assert row["_top2_prob"] == pytest.approx(0.2 / 0.7)
    assert "_top3_crate" not in row.index
    assert row["_allowed_crates"] == ["House%%Club", "House%%Deep"]


@pytest.mark.parametrize(
    ("model", "allowed", "expected"),
    [
        (FixedModel(), set(), "No crates are available"),
        (FixedModel((0.0, 0.0, 0.0)), {"House%%Club"}, "must total more than zero"),
        (FailingModel(), {"House%%Club"}, "predict_proba exploded"),
    ],
)
def test_prediction_failures_are_rows_not_crashes(
    prediction_dependencies, model, allowed, expected: str,
) -> None:
    rows, failures = propose_crates_for_files(
        {"model": model, "feature_columns": ["feature"]},
        [Path("/tmp/problem.mp3")],
        topk=1,
        identify_genre=False,
        allowed_crates=allowed,
    )
    assert rows.empty
    assert expected in failures.iloc[0]["error"]


def test_prediction_handles_missing_file_and_partial_failure(prediction_dependencies) -> None:
    calls = iter([[120.0], RuntimeError("file disappeared")])

    def feature_result(_path):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("phase4_engine.extract_audio_features", side_effect=feature_result):
        rows, failures = propose_crates_for_files(
            {"model": FixedModel(), "feature_columns": ["feature"]},
            [Path("/tmp/good.mp3"), Path("/tmp/missing.mp3")],
            identify_genre=False,
        )
    assert len(rows) == 1
    assert len(failures) == 1
    assert "file disappeared" in failures.iloc[0]["error"]


def test_embedded_metadata_is_used_before_local_genre_model(monkeypatch) -> None:
    with patch("genre_model.predict_genre") as local_model:
        result = identify_genre_year(Path("/tmp/tagged.mp3"), [1.0], {"genre": "Tag", "year": "1999"})
    assert result["genre"] == "Tag"
    assert result["year"] == "1999"
    assert result["genre_source"] == result["year_source"] == "embedded_tag"
    assert result["online_lookup_attempted"] is False
    local_model.assert_not_called()


def test_missing_year_stays_blank_when_local_genre_is_unavailable() -> None:
    with patch("genre_model.predict_genre", return_value=(None, 0.0)):
        result = identify_genre_year(Path("/tmp/no-tags.mp3"), [1.0], {"genre": "", "year": "not-a-date"})
    assert result["genre"] == ""
    assert result["year"] == ""
    assert result["year_source"] == "missing"
    assert result["online_lookup_attempted"] is False


def test_quality_thresholds_apply_after_allowed_crate_filtering_and_preserve_review_first_behavior(prediction_dependencies) -> None:
    bundle = {
        "model": FixedModel((0.55, 0.52, 0.12)), "feature_columns": ["feature"],
        "prediction_semantics": "independent_multilabel",
        "quality_configuration": {
            "threshold_configuration": {
                "global_threshold": 0.60, "per_crate": [["House%%Club", 0.60]],
                "minimum_threshold": 0.20, "maximum_threshold": 0.80, "minimum_support": 3,
                "source_split": "validation",
            },
            "training_support": {"House%%Club": 9, "Hip Hop%%Open Format": 9, "House%%Deep": 9},
            "low_confidence_probability": 0.40, "low_confidence_margin": 0.05,
            "per_crate_minimum_support": 2,
        },
    }
    rows, failures = propose_crates_for_files(
        bundle, [Path("/tmp/quality.mp3")], topk=5,
        allowed_crates={"House%%Club", "House%%Deep"},
    )
    assert failures.empty
    row = rows.iloc[0]
    assert row["Suggested Crate"] == "House%%Club"
    assert row["Confidence"] == pytest.approx(0.55)
    assert bool(row["Needs Review"])
    assert "threshold" in row["Review Reason"]
    assert row["Threshold Used"] == pytest.approx(0.60)
    assert row["_allowed_crates"] == ["House%%Club", "House%%Deep"]
