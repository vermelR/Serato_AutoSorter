"""Offline-first metadata tests: no provider makes a live request."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serato_ai.core.metadata_rules import METADATA_STRATEGY_TEXT, normalize_genre, normalize_year
from serato_ai.core.models import TrackTrainingRecord
from serato_ai.infrastructure.application_data import ApplicationDataPaths
from serato_ai.infrastructure.feature_cache_store import FeatureCacheStore
from serato_ai.infrastructure.metadata_providers import EmbeddedTagProvider, LocalGenreProvider
from serato_ai.services.feature_extraction_service import FeatureExtractionService
from serato_ai.services.metadata_enrichment_service import MetadataEnrichmentService


pytestmark = [pytest.mark.unit, pytest.mark.metadata]


def _service(tags: dict[str, str], prediction=("", 0.0)) -> tuple[MetadataEnrichmentService, MagicMock]:
    local = MagicMock(side_effect=lambda _features: prediction)
    return (
        MetadataEnrichmentService(
            embedded_provider=EmbeddedTagProvider(lambda _path: tags),
            local_genre_provider=LocalGenreProvider(local),
            minimum_local_confidence=0.5,
        ),
        local,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2024", "2024"), ("2024-05-10", "2024"), ("2024/05/10", "2024"), ("2024 May", "2024")],
)
def test_year_normalization_accepts_valid_tag_prefixes(raw: str, expected: str) -> None:
    assert normalize_year(raw, current_year=2026) == (expected, "")


@pytest.mark.parametrize("raw", ["", "0000", "3029-01-01", "unknown", "20xx", "2024bad"])
def test_year_normalization_rejects_invalid_or_unreasonable_values(raw: str) -> None:
    assert normalize_year(raw, current_year=2026)[0] == ""


def test_embedded_genre_and_year_win_without_local_or_online_lookup() -> None:
    service, local = _service({"title": "Song", "artist": "Artist", "genre": "  Deep House ; Disco  ", "year": "2024-05-10"}, ("Wrong", 0.99))
    result = service.enrich(Path("/tmp/song.mp3"), features=(1.0, 2.0))

    assert result.genre == "Deep House / Disco"
    assert result.genre_source == "embedded_tag"
    assert result.genre_confidence == 1.0
    assert result.year == "2024" and result.year_source == "embedded_tag"
    assert result.online_lookup_attempted is False
    assert result.provider_status[-2:] == ("acoustid:disabled", "musicbrainz:disabled")
    local.assert_not_called()


def test_missing_genre_uses_local_model_but_missing_year_stays_blank() -> None:
    service, local = _service({"genre": "", "year": ""}, ("Nu Disco", 0.75))
    result = service.enrich(Path("/tmp/song.mp3"), features=(1.0,))

    assert result.genre == "Nu Disco"
    assert result.genre_source == "local_genre_model"
    assert result.genre_confidence == pytest.approx(0.75)
    assert result.year == "" and result.year_source == "missing"
    assert result.online_lookup_attempted is False
    local.assert_called_once()


def test_low_confidence_or_invalid_embedded_genre_stays_unknown_without_overwriting_tags() -> None:
    service, local = _service({"genre": "unknown", "year": "2099"}, ("House", 0.2))
    result = service.enrich(Path("/tmp/song.mp3"), features=(1.0,))

    assert result.genre == "" and result.genre_source == "missing"
    assert result.year == "" and result.year_source == "missing"
    assert any("below the accepted threshold" in warning for warning in result.warnings)
    assert normalize_genre("R&B / UK Garage")[0] == "R&B / UK Garage"
    local.assert_called_once()


def test_feature_extraction_reads_tags_before_audio_and_records_metadata_without_editing_source(tmp_path: Path) -> None:
    audio = tmp_path / "Artist - Tagged Song.mp3"
    before = b"disposable source bytes"
    audio.write_bytes(before)
    order: list[str] = []
    metadata = MetadataEnrichmentService(
        embedded_provider=EmbeddedTagProvider(lambda _path: order.append("tags") or {"title": "Tagged", "artist": "Artist", "genre": "House", "year": "2023"}),
        local_genre_provider=LocalGenreProvider(lambda _features: ("Other", 0.99)),
    )
    extraction = FeatureExtractionService(
        cache=FeatureCacheStore(ApplicationDataPaths(tmp_path / "app-data")),
        extractor=lambda _path: order.append("audio") or [1.0, 2.0], metadata_service=metadata,
    )
    record = TrackTrainingRecord("id", str(audio), "", ("House%%Club",))
    values, _, extracted, failed, _ = extraction.extract((record,))

    assert order == ["tags", "audio"]
    assert extracted == 1 and failed == 0
    assert values[0].genre == "House" and values[0].genre_source == "embedded_tag"
    assert values[0].year == "2023" and values[0].year_source == "embedded_tag"
    assert audio.read_bytes() == before
    assert METADATA_STRATEGY_TEXT.startswith("Existing tags first")
