from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from serato_ai.services.prediction_service import PredictionService
from serato_ai.services.watcher_service import WatcherService
from watcher import Watcher


pytestmark = [pytest.mark.unit, pytest.mark.watcher]


def test_watcher_service_injects_the_shared_prediction_service() -> None:
    factory = MagicMock(return_value=object())
    prediction = PredictionService(file_collector=lambda *_args: [], predictor=lambda *_args, **_kwargs: (None, None))
    watcher = WatcherService(prediction_service=prediction, watcher_factory=factory).create(
        folders=["/tmp/music"], crates=[], model_path="model.pkl", topk=3,
        identify_genre=True, excluded_crates={"House%%Deep"}, allowed_crates={"House%%Club"},
    )
    assert watcher is factory.return_value
    assert factory.call_args.kwargs["prediction_service"] is prediction


def test_watcher_service_updates_filters_without_ui_imports() -> None:
    watcher = MagicMock()
    WatcherService.update_filters(watcher, allowed_crates={"House%%Club"}, excluded_crates=set())
    watcher.set_crate_filters.assert_called_once_with(
        allowed_crates={"House%%Club"}, excluded_crates=set(),
    )


def test_watcher_uses_injected_prediction_service_for_the_same_rules_as_manual(monkeypatch) -> None:
    prediction = MagicMock()
    prediction.predict_files.return_value = object()
    prediction.to_dataframes.return_value = (
        pd.DataFrame([{
            "path": "/tmp/watched.mp3", "Suggested Crate": "House%%Club", "Confidence": 1.0,
            "_top1_crate": "House%%Club", "_top1_prob": 1.0,
            "Prediction Quality": "Review Recommended", "Needs Review": True,
            "Review Reason": "No crate exceeded its configured decision threshold.",
            "Top Margin": 0.01, "Threshold Used": 0.6, "Supported Crate Count": 4,
        }]),
        pd.DataFrame(),
    )
    watcher = Watcher(folders=[], allowed_crates={"House%%Club"}, prediction_service=prediction)
    watcher._bundle = {"model": object()}
    watcher._candidate_files = lambda: [Path("/tmp/watched.mp3")]  # type: ignore[method-assign]
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]
    queued: list[dict] = []
    monkeypatch.setattr("watcher.append_pending", queued.append)

    watcher._scan_once()

    assert prediction.predict_files.call_args.kwargs["allowed_crates"] == {"House%%Club"}
    assert prediction.predict_files.call_args.kwargs["excluded_crates"] == set()
    assert queued[0]["Review Reason"] == "No crate exceeded its configured decision threshold."
    assert queued[0]["Threshold Used"] == 0.6
