"""Safe, minimal helpers for Serato's binary crate and database records.

Serato crate files are binary containers.  The project only needs the track
path stored in each ``otrk``/``ptrk`` record, so this module deliberately
parses that field alone rather than decoding an entire file as text.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class CrateTrackEntry:
    """One parsed path reference and its precise location in the file."""

    path: str
    serialized_path: str
    field_start: int
    field_end: int
    encoding: str


class SeratoCrateParseError(ValueError):
    """A crate/database record cannot be safely decoded or updated."""

    def __init__(self, file_path: str | Path, start: int, end: int, detail: str):
        self.file_path = str(file_path)
        self.start = start
        self.end = end
        self.detail = detail
        super().__init__(
            f"Malformed Serato crate data in {self.file_path} at bytes "
            f"[{self.start}:{self.end}): {self.detail}"
        )


def _decode_utf16_field(
    payload: bytes,
    *,
    file_path: str | Path,
    start: int,
    end: int,
) -> tuple[str, str]:
    """Decode one validated Serato text field, never an entire binary file."""
    if len(payload) % 2:
        raise SeratoCrateParseError(
            file_path,
            start,
            end,
            "Malformed Serato crate field: expected an even UTF-16 byte length, "
            f"received {len(payload)} byte{'s' if len(payload) != 1 else ''}",
        )
    if not payload:
        # An empty field is valid UTF-16 data.  Preserve it as an empty value;
        # callers can decide whether it is meaningful for their record type.
        return "", "utf-16-be"

    # Existing Serato files in this library use Java writeChars-style UTF-16BE
    # (``00 2f`` for a leading slash).  Accept explicit UTF-16LE fields too so
    # malformed/legacy input gets a precise result instead of a decoder crash.
    if payload.startswith(b"\xff\xfe") or payload.startswith(b"/\x00"):
        encoding = "utf-16-le"
    else:
        encoding = "utf-16-be"

    try:
        value = payload.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SeratoCrateParseError(
            file_path,
            start + exc.start,
            start + max(exc.end, exc.start + 1),
            f"Invalid {encoding} field: {exc.reason}",
        ) from exc
    value = value.lstrip("\ufeff")
    # A path field is text, not an arbitrary even-length binary blob.  UTF-16
    # can technically decode NUL/control bytes and isolated surrogates, but
    # accepting those values would turn corruption into a bogus filesystem
    # path and make a subsequent crate update unsafe.
    if any(ord(character) < 32 or unicodedata.category(character) == "Cs" for character in value):
        raise SeratoCrateParseError(
            file_path,
            start,
            end,
            f"Invalid {encoding} field: non-text control or surrogate character",
        )
    return value, encoding


def _parse_record_paths(
    data: bytes,
    *,
    file_path: str | Path,
    field_marker: bytes,
    container_name: str,
) -> list[CrateTrackEntry]:
    """Parse length-bounded path fields nested in ``otrk`` records."""
    entries: list[CrateTrackEntry] = []
    search_from = 0

    while True:
        record_start = data.find(b"otrk", search_from)
        if record_start < 0:
            break
        record_header_end = record_start + 8
        if record_header_end > len(data):
            raise SeratoCrateParseError(
                file_path,
                record_start,
                len(data),
                f"Truncated {container_name} otrk record header",
            )

        record_size = int.from_bytes(data[record_start + 4:record_header_end], "big")
        record_end = record_header_end + record_size
        if record_end > len(data):
            raise SeratoCrateParseError(
                file_path,
                record_start,
                len(data),
                f"Truncated {container_name} otrk record: declared {record_size} bytes",
            )

        field_start = data.find(field_marker, record_header_end, record_end)
        if field_start < 0:
            raise SeratoCrateParseError(
                file_path,
                record_header_end,
                record_end,
                f"otrk record does not contain a {field_marker.decode('latin1')} path field",
            )
        field_header_end = field_start + 8
        if field_header_end > record_end:
            raise SeratoCrateParseError(
                file_path,
                field_start,
                record_end,
                f"Truncated {field_marker.decode('latin1')} field header",
            )

        field_size = int.from_bytes(data[field_start + 4:field_header_end], "big")
        value_start = field_header_end
        value_end = value_start + field_size
        if value_end > record_end:
            raise SeratoCrateParseError(
                file_path,
                value_start,
                record_end,
                f"Truncated {field_marker.decode('latin1')} field: declared {field_size} bytes",
            )

        serialized_path, encoding = _decode_utf16_field(
            data[value_start:value_end],
            file_path=file_path,
            start=value_start,
            end=value_end,
        )
        # Serato's on-disk ``ptrk``/``pfil`` fields conventionally omit the
        # leading POSIX slash (``Users/...``).  Restore it for filesystem and
        # comparison use while retaining the raw representation for diagnostics.
        path = serialized_path if serialized_path.startswith("/") or not serialized_path else f"/{serialized_path}"
        entries.append(CrateTrackEntry(path, serialized_path, value_start, value_end, encoding))
        search_from = record_end

    return entries


def parse_serato_crate_bytes(data: bytes, file_path: str | Path = "<memory>") -> list[CrateTrackEntry]:
    """Return track paths from a binary ``.crate`` file after strict checks."""
    if not data:
        raise SeratoCrateParseError(file_path, 0, 0, "Serato crate file is empty")
    if not data.startswith(b"vrsn"):
        raise SeratoCrateParseError(file_path, 0, min(len(data), 4), "Missing Serato vrsn header")
    return _parse_record_paths(
        data,
        file_path=file_path,
        field_marker=b"ptrk",
        container_name="crate",
    )


def parse_serato_crate(file_path: str | Path) -> list[CrateTrackEntry]:
    """Read a crate in binary mode and parse only its length-bounded paths."""
    path = Path(file_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SeratoCrateParseError(path, 0, 0, f"Could not read crate file: {exc}") from exc
    return parse_serato_crate_bytes(data, path)


def parse_serato_database_paths(file_path: str | Path) -> list[CrateTrackEntry]:
    """Read ``database V2`` paths for diagnostics only; this never writes it."""
    path = Path(file_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SeratoCrateParseError(path, 0, 0, f"Could not read database file: {exc}") from exc
    if not data:
        return []
    if not data.startswith(b"vrsn"):
        raise SeratoCrateParseError(path, 0, min(len(data), 4), "Missing Serato vrsn header")
    return _parse_record_paths(
        data,
        file_path=path,
        field_marker=b"pfil",
        container_name="database",
    )


def serialize_serato_track_record(serialized_path: str) -> bytes:
    """Create one valid ``otrk``/``ptrk`` reference using Serato's UTF-16BE."""
    encoded_path = serialized_path.encode("utf-16-be")
    return (
        b"otrk"
        + (len(encoded_path) + 8).to_bytes(4, "big")
        + b"ptrk"
        + len(encoded_path).to_bytes(4, "big")
        + encoded_path
    )


def serato_serialized_track_path(path: Path) -> str:
    """Return the POSIX path spelling used in Serato's ptrk/pfil fields."""
    serialized = str(path)
    return serialized[1:] if serialized.startswith("/") else serialized


def _normalized_input_path(value: str | Path) -> str:
    raw = unquote(str(value).strip())
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Unsupported non-local file URL: {value}")
        raw = unquote(parsed.path)
    return unicodedata.normalize("NFC", raw)


def canonical_track_path(value: str | Path) -> tuple[Path, str]:
    """Return the exact path serialized to Serato plus a stable comparison key."""
    path = Path(_normalized_input_path(value)).expanduser().resolve()
    serialized = str(path)
    # macOS installations are generally case-insensitive.  Use casefold in
    # addition to normcase so comparisons stay safe when tests run elsewhere.
    comparison_key = os.path.normcase(unicodedata.normalize("NFC", serialized)).casefold()
    return path, comparison_key


def canonical_crate_name(value: object) -> tuple[str, str]:
    """Keep Serato's raw name for filenames and a stable key for comparisons."""
    raw_name = unicodedata.normalize("NFC", str(value).strip())
    return raw_name, raw_name.casefold()
