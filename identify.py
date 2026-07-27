"""Deprecated online-identification compatibility shim.

SeratoAI M4 deliberately performs no AcoustID, Chromaprint, or MusicBrainz
requests. Future software can implement those capabilities through
``serato_ai.infrastructure.metadata_providers`` without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentifyResult:
    genre: str = ""
    year: str = ""
    artist: str = ""
    title: str = ""
    score: float = 0.0
    mb_recording_id: str = ""
    source: str = "disabled"


def lookup_by_fingerprint(_file_path: str) -> None:
    """Return no result: live online metadata is disabled in this release."""
    return None
