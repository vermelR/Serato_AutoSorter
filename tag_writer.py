"""
Write genre/year metadata into the audio file itself.

Serato does not keep its own separate genre/year database — it reads the
Genre and Year columns straight from each file's tags (ID3 for mp3, Vorbis
comments for flac, MP4 atoms for m4a, etc). So "adding the year/genre in
Serato" means writing it into the file's tags; mutagen's "easy" interface
normalizes the tag name across most formats, so this mostly doesn't need
per-format logic. The exception is WAV/AIFF: mutagen's `add_tags()` for
those containers creates a raw ID3 tag object (not the dict-like "Easy"
wrapper) whenever a file has no existing tags, so we fall back to setting
ID3 frames directly for those.
"""

from __future__ import annotations

from mutagen import File as MutagenFile
from mutagen.id3 import TCON, TDRC

from serato_ai.core.models import TagWriteResult


def write_genre_year(path: str, genre: str = "", year: str = "") -> TagWriteResult:
    genre = (genre or "").strip()
    year = (year or "").strip()

    if not genre and not year:
        return TagWriteResult(path, True, "nothing to write")

    try:
        audio = MutagenFile(path, easy=True)
        if audio is None:
            return TagWriteResult(path, False, "Unsupported/unrecognized audio format")

        if audio.tags is None:
            audio.add_tags()

        try:
            if genre:
                audio.tags["genre"] = [genre]
            if year:
                audio.tags["date"] = [year]
        except TypeError:
            # WAV/AIFF: add_tags() gave us a raw ID3 container instead of an
            # Easy-mapped one. Set the equivalent frames directly.
            if genre:
                audio.tags.setall("TCON", [TCON(encoding=3, text=[genre])])
            if year:
                audio.tags.setall("TDRC", [TDRC(encoding=3, text=[year])])

        audio.save()
        return TagWriteResult(path, True)

    except Exception as e:
        return TagWriteResult(path, False, f"{type(e).__name__}: {e}")
