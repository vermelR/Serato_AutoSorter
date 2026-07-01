"""
Identify the *original* recording behind an audio file using acoustic
fingerprinting (AcoustID/Chromaprint), then pull that recording's genre and
release year from MusicBrainz.

This is the "based on how it sounds and the original song" part: instead of
trusting whatever (possibly wrong/missing) ID3 tags a downloaded file has,
we fingerprint the actual audio, match it against AcoustID's database of
known recordings, and read the real genre/year from MusicBrainz.

Falls back cleanly (returns None) if:
- no ACOUSTID_API_KEY is configured
- `fpcalc` (chromaprint) isn't installed/bundled
- no confident match is found
- the network/API is unavailable

Callers should treat a None result as "no fingerprint match" and fall back
to the local ML genre classifier and/or existing file tags.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

try:
    import acoustid
except ImportError:
    acoustid = None

try:
    import musicbrainzngs
except ImportError:
    musicbrainzngs = None


_MB_CONFIGURED = False


@dataclass
class IdentifyResult:
    genre: str = ""
    year: str = ""
    artist: str = ""
    title: str = ""
    score: float = 0.0
    mb_recording_id: str = ""
    source: str = "fingerprint"


def _bundled_fpcalc_path() -> Optional[Path]:
    p = (Path(__file__).parent / "vendor" / "fpcalc").resolve()
    return p if p.exists() else None


def _ensure_musicbrainz_configured() -> None:
    global _MB_CONFIGURED
    if _MB_CONFIGURED or musicbrainzngs is None:
        return
    musicbrainzngs.set_useragent(
        config.MB_APP_NAME, config.MB_APP_VERSION, config.MB_CONTACT_EMAIL
    )
    _MB_CONFIGURED = True


def _pick_genre_from_tags(tags: list[dict]) -> str:
    """MusicBrainz 'tags' are community folksonomy tags with vote counts.
    Not every tag is a genre (could be a mood/decade/etc), but picking the
    highest-voted tag is a reasonable heuristic and is what most tools do."""
    if not tags:
        return ""
    ranked = sorted(tags, key=lambda t: int(t.get("count", 0)), reverse=True)
    return ranked[0]["name"].title()


def _pick_earliest_year(release_list: list[dict]) -> str:
    years = []
    for rel in release_list or []:
        date = rel.get("date", "")
        if date and date[:4].isdigit():
            years.append(int(date[:4]))
    if not years:
        return ""
    return str(min(years))


def lookup_by_fingerprint(file_path: str) -> Optional[IdentifyResult]:
    """Fingerprint `file_path` and identify the original recording via
    AcoustID + MusicBrainz. Returns None if unavailable/no confident match."""

    if not config.ACOUSTID_API_KEY:
        return None
    if acoustid is None or musicbrainzngs is None:
        return None

    fpcalc = _bundled_fpcalc_path()
    if fpcalc is not None:
        acoustid.FPCALC_COMMAND = str(fpcalc)

    try:
        matches = list(
            acoustid.match(config.ACOUSTID_API_KEY, file_path, parse=True)
        )
    except Exception:
        # Covers missing fpcalc backend, network errors, bad API key, etc.
        return None

    if not matches:
        return None

    # acoustid.match yields (score, recording_id, title, artist) sorted by score
    score, recording_id, title, artist = matches[0]
    if score is None or score < config.ACOUSTID_MIN_SCORE or not recording_id:
        return None

    _ensure_musicbrainz_configured()

    try:
        mb = musicbrainzngs.get_recording_by_id(
            recording_id, includes=["tags", "releases", "artist-credits"]
        )
    except Exception:
        # Fingerprint matched but MusicBrainz lookup failed (offline, rate
        # limited, etc). Still return what AcoustID gave us.
        return IdentifyResult(
            artist=artist or "",
            title=title or "",
            score=float(score),
            mb_recording_id=recording_id,
        )

    recording = mb.get("recording", {})
    genre = _pick_genre_from_tags(recording.get("tag-list", []))
    year = _pick_earliest_year(recording.get("release-list", []))

    return IdentifyResult(
        genre=genre,
        year=year,
        artist=artist or "",
        title=title or "",
        score=float(score),
        mb_recording_id=recording_id,
    )
