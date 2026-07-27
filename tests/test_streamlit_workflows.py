from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from serato_writer import SeratoWriteResult
from tag_writer import TagWriteResult
from serato_ai.infrastructure.onboarding_store import OnboardingStore
from watcher import append_pending_many, load_pending_queue


pytestmark = pytest.mark.streamlit


CRATES = ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]


def _prediction(*, approve: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Song Title": "Test",
                "Artist": "Artist",
                "path": "/tmp/test.mp3",
                "Approve": approve,
                "Suggested Crate": "House%%Club",
                "Confidence": 0.825,
                "_top1_crate": "House%%Club",
                "_top1_prob": 0.825,
                "_top2_crate": "Hip Hop%%Open Format",
                "_top2_prob": 0.09,
                "_top3_crate": "House%%Deep",
                "_top3_prob": 0.085,
            }
        ]
    )


def _button(app: AppTest, label: str):
    return next(widget for widget in app.button if widget.label == label)


def test_first_launch_routes_to_safe_onboarding() -> None:
    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    assert not app.exception
    assert "Start setup" in {widget.label for widget in app.button}
    assert "🔮 Generate Predictions (manual scan)" not in {widget.label for widget in app.button}


def test_start_setup_persists_the_locations_entered_by_the_dj() -> None:
    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    root = "/temporary/custom/_Serato_"
    folders = "/temporary/music one\n/temporary/music two"
    next(widget for widget in app.text_input if widget.label == "Serato library folder").set_value(root)
    next(widget for widget in app.text_area if widget.label == "Music folders (one per line)").set_value(folders)
    _button(app, "Start setup").click()
    app.run(timeout=30)
    status = OnboardingStore().status()
    assert status.phase == "in_progress"
    assert status.serato_root == root
    assert status.music_folders == ("/temporary/music one", "/temporary/music two")


def test_manual_final_crates_render_persist_and_do_not_double_confidence() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction()
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    app.run(timeout=30)

    picker = next(widget for widget in app.multiselect if widget.label.startswith("Final Crates:"))
    assert picker.options == ["House › Club", "Hip Hop › Open Format", "House › Deep"]
    assert app.session_state["pred_df"].iloc[0]["Final Crates"] == ["House%%Club"]
    assert app.session_state["pred_df"].iloc[0]["Confidence"] == pytest.approx(0.825)
    assert app.session_state["pred_df"].iloc[0]["Confidence (%)"] == pytest.approx(82.5)

    app.run(timeout=30)
    assert app.session_state["pred_df"].iloc[0]["Confidence"] == pytest.approx(0.825)
    assert app.session_state["pred_df"].iloc[0]["Confidence (%)"] == pytest.approx(82.5)
    assert app.session_state["pred_df"].iloc[0]["Final Crates"] == ["House%%Club"]


def test_queue_pagination_search_and_final_crates_do_not_repeat_expensive_work() -> None:
    rows = [
        {
            "Song Title": f"Queued {index}",
            "Artist": "Fixture",
            "path": f"/tmp/queued-{index}.mp3",
            "Approve": False,
            "Suggested Crate": "House%%Club",
            "Confidence": 0.825,
            "Needs Review": index % 2 == 0,
            "_top1_crate": "House%%Club",
            "_top1_prob": 0.825,
            "_top2_crate": "House%%Deep",
            "_top2_prob": 0.125,
            "_allowed_crates": CRATES,
        }
        for index in range(120)
    ]
    assert sum(append_pending_many(rows)) == len(rows)
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES

    with (
        patch("serato_ai.services.model_service.ModelService.load") as load,
        patch(
            "serato_ai.services.library_scan_service.LibraryScanService.scan"
        ) as scan,
        patch(
            "serato_ai.services.training_service.TrainingService.train"
        ) as train,
        patch(
            "serato_ai.services.prediction_service.PredictionService.predict_files"
        ) as predict,
    ):
        app.run(timeout=30)
        assert len(
            [
                widget
                for widget in app.multiselect
                if widget.label.startswith("Final Crates:")
            ]
        ) == 50

        next(
            widget for widget in app.number_input if widget.label == "Page"
        ).set_value(2)
        app.run(timeout=30)
        search = next(
            widget
            for widget in app.text_input
            if widget.label == "Search predictions"
        )
        search.set_value("Queued 119")
        app.run(timeout=30)
        picker = next(
            widget
            for widget in app.multiselect
            if widget.label.startswith("Final Crates:")
        )
        picker.set_value(["House%%Club", "House%%Deep"])
        app.run(timeout=30)

    load.assert_not_called()
    scan.assert_not_called()
    train.assert_not_called()
    predict.assert_not_called()
    saved = load_pending_queue(search="Queued 119")
    assert len(saved) == 1
    assert saved[0]["Final Crates"] == ["House%%Club", "House%%Deep"]


def test_manual_approve_state_rerun_does_not_scan_extract_train_or_evaluate() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction()
    app.session_state["manual_prediction_filter_signature"] = tuple(
        sorted(CRATES)
    )
    with (
        patch("serato_ai.services.model_service.ModelService.load") as load,
        patch(
            "serato_ai.services.library_scan_service.LibraryScanService.scan"
        ) as scan,
        patch(
            "serato_ai.services.feature_extraction_service.FeatureExtractionService.extract"
        ) as extract,
        patch(
            "serato_ai.services.training_service.TrainingService.train"
        ) as train,
        patch(
            "serato_ai.services.model_quality_service.ModelQualityService.evaluate"
        ) as evaluate,
    ):
        app.run(timeout=30)
        frame = app.session_state["pred_df"]
        frame.at[frame.index[0], "Approve"] = True
        app.session_state["pred_df"] = frame
        app.run(timeout=30)

    assert bool(app.session_state["pred_df"].iloc[0]["Approve"])
    load.assert_not_called()
    scan.assert_not_called()
    extract.assert_not_called()
    train.assert_not_called()
    evaluate.assert_not_called()


def test_apply_without_predictions_shows_validation_warning() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.run(timeout=30)
    _button(app, "✅ Apply APPROVED to Serato").click()
    app.run(timeout=30)
    assert any("Generate predictions first" in warning.value for warning in app.warning)


def test_standard_metadata_setting_is_offline_first_and_has_no_api_key_field() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.run(timeout=30)
    labels = {widget.label for widget in app.sidebar.checkbox}
    assert "Identify genre and year automatically" in labels
    assert "AcoustID API key" not in {widget.label for widget in app.sidebar.text_input}
    setting = next(widget for widget in app.sidebar.checkbox if widget.label == "Identify genre and year automatically")
    assert setting.value is True


def test_approved_track_without_final_crates_is_blocked_before_writer_call() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction(approve=True)
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    app.session_state["manual_crate_selections"] = {"/tmp/test.mp3": []}
    with patch("serato_writer.write_tracks_to_crates") as writer:
        app.run(timeout=30)
        _button(app, "✅ Apply APPROVED to Serato").click()
        app.run(timeout=30)
    writer.assert_not_called()
    assert any("Select at least one Final Crate" in warning.value for warning in app.warning)


def test_live_success_message_requires_all_assignment_results(temporary_serato_root: Path) -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction(approve=True)
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    with (
        patch("serato_writer.write_tracks_to_crates", return_value=[
            SeratoWriteResult("House%%Club", "/tmp/test.mp3", True, True, "added")
        ]) as writer,
        patch("tag_writer.write_genre_year", return_value=TagWriteResult("/tmp/test.mp3", True)),
    ):
        app.run(timeout=30)
        root_input = next(
            widget for widget in app.sidebar.text_input if widget.label == "Serato folder (root)"
        )
        root_input.set_value(str(temporary_serato_root))
        dry_run = next(widget for widget in app.sidebar.checkbox if widget.label == "Dry run (no changes)")
        dry_run.set_value(False)
        app.run(timeout=30)
        _button(app, "✅ Apply APPROVED to Serato").click()
        app.run(timeout=30)
    writer.assert_called_once()
    assert any("All 1 Serato crate assignment" in success.value for success in app.success)


def test_partial_failure_uses_warning_not_success_banner(temporary_serato_root: Path) -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction(approve=True)
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    with (
        patch("serato_writer.write_tracks_to_crates", return_value=[
            SeratoWriteResult("House%%Club", "/tmp/test.mp3", True, True, "added"),
            SeratoWriteResult("House%%Deep", "/tmp/test.mp3", False, False, "failed", "denied"),
        ]),
        patch("tag_writer.write_genre_year", return_value=TagWriteResult("/tmp/test.mp3", True)),
    ):
        app.run(timeout=30)
        next(widget for widget in app.sidebar.text_input if widget.label == "Serato folder (root)").set_value(
            str(temporary_serato_root)
        )
        next(widget for widget in app.sidebar.checkbox if widget.label == "Dry run (no changes)").set_value(False)
        app.run(timeout=30)
        _button(app, "✅ Apply APPROVED to Serato").click()
        app.run(timeout=30)
    assert any("1 of 2 Serato crate assignment" in warning.value for warning in app.warning)
    assert not any("All 2 Serato crate assignment" in success.value for success in app.success)


def test_all_failed_assignment_results_use_error_not_success_banner(temporary_serato_root: Path) -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction(approve=True)
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    with (
        patch("serato_writer.write_tracks_to_crates", return_value=[
            SeratoWriteResult("House%%Club", "/tmp/test.mp3", False, False, "failed", "denied")
        ]),
        patch("tag_writer.write_genre_year", return_value=TagWriteResult("/tmp/test.mp3", True)),
    ):
        app.run(timeout=30)
        next(widget for widget in app.sidebar.text_input if widget.label == "Serato folder (root)").set_value(
            str(temporary_serato_root)
        )
        next(widget for widget in app.sidebar.checkbox if widget.label == "Dry run (no changes)").set_value(False)
        app.run(timeout=30)
        _button(app, "✅ Apply APPROVED to Serato").click()
        app.run(timeout=30)
    assert any("No Serato crate assignments succeeded" in error.value for error in app.error)
    assert not any("All 1 Serato crate assignment" in success.value for success in app.success)


def test_dry_run_apply_shows_informational_message_and_does_not_call_tag_writer() -> None:
    app = AppTest.from_file("app.py")
    app.session_state["crate_options"] = CRATES
    app.session_state["pred_df"] = _prediction(approve=True)
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(CRATES))
    with (
        patch("serato_writer.write_tracks_to_crates", return_value=[
            SeratoWriteResult("House%%Club", "/tmp/test.mp3", True, False, "dry_run")
        ]),
        patch("tag_writer.write_genre_year") as tag_writer,
    ):
        app.run(timeout=30)
        _button(app, "✅ Apply APPROVED to Serato").click()
        app.run(timeout=30)
    tag_writer.assert_not_called()
    assert any("Dry run is ON" in info.value for info in app.info)
