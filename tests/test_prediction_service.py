from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from serato_ai.core.models import PredictionRequest
from serato_ai.services.model_service import ModelService
from serato_ai.services.prediction_service import PredictionService


pytestmark = [pytest.mark.unit, pytest.mark.prediction]


class Model:
    classes_ = ["House%%Club", "Hip Hop%%Open"]


def test_prediction_service_uses_mocked_dependencies_and_preserves_raw_contracts() -> None:
    calls: dict[str, object] = {}
    bundle = {"model": Model(), "feature_columns": ["feature"]}

    def collect(paths: list[str], recursive: bool) -> list[Path]:
        calls["collect"] = (paths, recursive)
        return [Path("/tmp/song.mp3")]

    def predict(model_bundle, files, **kwargs):
        calls["predict"] = (model_bundle, files, kwargs)
        return (
            pd.DataFrame([{
                "path": "/tmp/song.mp3", "Song Title": "Song", "Suggested Crate": "House%%Club",
                "Confidence": 0.8, "_top1_crate": "House%%Club", "_top1_prob": 0.8,
                "Genre": "House", "Genre Source": "embedded_tag", "Genre Confidence": 1.0,
                "Year": "2024", "Year Source": "embedded_tag", "_metadata_raw_year": "2024-05-10",
                "_metadata_warnings": ["diagnostic"], "_online_lookup_attempted": False,
                "_metadata_provider_status": ["embedded_tags:available", "acoustid:disabled"],
                "Prediction Quality": "Review Recommended", "Needs Review": True,
                "Review Reason": "Top probabilities are close.", "Top Margin": 0.03,
                "Threshold Used": 0.6, "Supported Crate Count": 5,
                "_allowed_crates": ["House%%Club"],
            }]),
            pd.DataFrame(),
        )

    service = PredictionService(
        model_service=ModelService(loader=lambda _path: bundle), file_collector=collect, predictor=predict,
    )
    response = service.predict(PredictionRequest(
        ("/tmp/music",), "fixture.pkl", topk=5, recursive=False,
        identify_genre=False, allowed_crates=("House%%Club",), excluded_crates=("Hip Hop%%Open",),
    ))

    assert calls["collect"] == (["/tmp/music"], False)
    assert calls["predict"][2] == {
        "topk": 5, "identify_genre": False, "allowed_crates": {"House%%Club"},
        "excluded_crates": {"Hip Hop%%Open"},
    }
    assert response.predictions[0].suggested_crate == "House%%Club"
    assert response.predictions[0].suggestions[0].crate_name == "House%%Club"
    assert response.predictions[0].suggestions[0].probability == pytest.approx(0.8)
    assert response.predictions[0].raw_year == "2024-05-10"
    assert response.predictions[0].metadata_warnings == ("diagnostic",)
    assert response.predictions[0].online_lookup_attempted is False
    assert response.predictions[0].provider_status == ("embedded_tags:available", "acoustid:disabled")
    assert response.predictions[0].needs_review
    assert response.predictions[0].review_reason == "Top probabilities are close."
    assert response.predictions[0].threshold_used == pytest.approx(0.6)
    frame, failures = service.to_dataframes(response)
    assert frame.iloc[0]["_top1_crate"] == "House%%Club"
    assert frame.iloc[0]["_metadata_provider_status"] == ["embedded_tags:available", "acoustid:disabled"]
    assert failures.empty


def test_prediction_service_never_imports_streamlit() -> None:
    # Importing a service can load dependencies but must never start the UI.
    assert "streamlit" not in inspect.getsource(PredictionService).lower()
