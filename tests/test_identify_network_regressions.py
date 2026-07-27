from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import identify
import genre_model
from serato_ai.infrastructure.metadata_providers import AcoustIDProvider, MusicBrainzProvider


pytestmark = [pytest.mark.unit, pytest.mark.prediction]


def test_legacy_fingerprint_entrypoint_is_disabled_without_network() -> None:
    assert identify.lookup_by_fingerprint("/tmp/song.mp3") is None


def test_future_online_provider_adapters_are_disabled_without_calling_network() -> None:
    for provider in (AcoustIDProvider(), MusicBrainzProvider()):
        result = provider.read(Path("/tmp/song.mp3"), (1.0,))
        assert result.status == "disabled"
        assert provider.name in result.warnings[0].casefold()


def test_local_genre_model_cache_is_not_reloaded_for_simple_reruns(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "genre-model.pkl"
    model_path.write_bytes(b"fixture")
    bundle = {"model": object(), "feature_columns": ["feature"]}
    monkeypatch.setattr(genre_model, "_genre_bundle_cache", None)
    with patch("genre_model.joblib.load", return_value=bundle) as load:
        first = genre_model.load_genre_model(str(model_path))
        second = genre_model.load_genre_model(str(model_path))
    assert first is second is bundle
    load.assert_called_once_with(str(model_path))
