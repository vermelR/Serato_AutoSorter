from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest

from serato_crate import SeratoCrateParseError, parse_serato_crate, parse_serato_crate_bytes
from serato_writer import (
    SeratoWriteResult,
    validate_serato_root,
    write_tracks_to_crates,
)
from tests.conftest import REAL_SERATO_ROOT


pytestmark = [pytest.mark.writer, pytest.mark.integration]


def _header() -> bytes:
    return b"vrsn\x00\x00\x00\x00"


def _record(payload: bytes, *, declared_size: int | None = None) -> bytes:
    size = len(payload) if declared_size is None else declared_size
    return b"otrk" + (len(payload) + 8).to_bytes(4, "big") + b"ptrk" + size.to_bytes(4, "big") + payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "crate file is empty"),
        (_header() + b"otrk", "Truncated crate otrk record header"),
        (_header() + b"otrk\x00\x00\x00\x20ptrk\x00\x00\x00\x00", "Truncated crate otrk record"),
        (_header() + _record(b"/", declared_size=2), "Truncated ptrk field"),
        (_header() + _record(b"\x00\x00"), "non-text control"),
        (_header() + b"otrk\x00\x00\x00\x08xxxx\x00\x00\x00\x00", "does not contain a ptrk"),
    ],
)
def test_binary_parser_rejects_malformed_structures_without_decoding_whole_file(payload, message) -> None:
    with pytest.raises(SeratoCrateParseError, match=message):
        parse_serato_crate_bytes(payload, "temporary-malformed.crate")


def test_malformed_file_is_byte_identical_after_writer_rejection(
    temporary_serato_root: Path, temporary_audio_file: Path, malformed_crate_file: Path,
) -> None:
    before = malformed_crate_file.read_bytes()
    result = write_tracks_to_crates(
        temporary_serato_root, [("Malformed", temporary_audio_file)], dry_run=False
    )[0]
    assert not result.success
    assert result.status == "failed"
    assert malformed_crate_file.read_bytes() == before


def test_string_and_path_inputs_have_identical_canonical_result_and_dataframe_shape(
    temporary_serato_root: Path, temporary_audio_file: Path,
) -> None:
    string_result = write_tracks_to_crates(
        temporary_serato_root, [("Strings", str(temporary_audio_file))], dry_run=True
    )[0]
    path_result = write_tracks_to_crates(
        temporary_serato_root, [("Paths", temporary_audio_file)], dry_run=True
    )[0]

    assert string_result.track_path == path_result.track_path == str(temporary_audio_file.resolve())
    assert string_result.status == path_result.status == "dry_run"
    assert set(string_result.__dict__) == {
        "crate_name", "track_path", "success", "changed", "status", "error", "backup_path",
    }
    assert not hasattr(str(temporary_audio_file), "path")


def test_duplicate_relative_absolute_space_and_unicode_assignments_are_written_once(
    temporary_serato_root: Path, temporary_audio_file: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(temporary_audio_file.parent)
    relative = Path(temporary_audio_file.name)
    nfd = unicodedata.normalize("NFD", str(temporary_audio_file))
    results = write_tracks_to_crates(
        temporary_serato_root,
        [
            ("Spaced Crate", relative),
            ("Spaced Crate", temporary_audio_file),
            ("Spaced Crate", str(temporary_audio_file)),
            ("Spaced Crate", nfd),
            ("Second Crate", str(temporary_audio_file)),
        ],
        dry_run=False,
        make_backup=False,
    )

    assert [(row.crate_name, row.status) for row in results] == [
        ("Spaced Crate", "added"),
        ("Second Crate", "added"),
    ]
    first = parse_serato_crate(temporary_serato_root / "SubCrates" / "Spaced Crate.crate")
    second = parse_serato_crate(temporary_serato_root / "SubCrates" / "Second Crate.crate")
    assert [entry.path for entry in first] == [str(temporary_audio_file.resolve())]
    assert [entry.path for entry in second] == [str(temporary_audio_file.resolve())]


@pytest.mark.parametrize(
    "root_kind",
    ["missing", "missing-subcrates", "backup-parent", "backup-snapshot"],
)
def test_invalid_serato_roots_return_readable_writer_failures(
    tmp_path: Path, temporary_audio_file: Path, root_kind: str,
) -> None:
    root = tmp_path / "candidate"
    if root_kind == "missing-subcrates":
        root.mkdir()
    elif root_kind == "backup-parent":
        root = tmp_path / "SeratoAI_Backups"
        (root / "SubCrates").mkdir(parents=True)
    elif root_kind == "backup-snapshot":
        root = tmp_path / "_Serato__backup_20260101_000000"
        (root / "SubCrates").mkdir(parents=True)

    result = write_tracks_to_crates(root, [("One", temporary_audio_file)], dry_run=True)[0]
    assert not result.success
    assert result.status == "failed"
    assert result.error


def test_valid_relative_space_unicode_and_home_expanded_roots_are_resolved(tmp_path: Path, monkeypatch) -> None:
    spaced = tmp_path / "root with spaces" / "café Serato"
    (spaced / "SubCrates").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert validate_serato_root(Path("root with spaces") / "café Serato") == spaced.resolve()

    fake_home = tmp_path / "fake-home"
    (fake_home / "Serato" / "SubCrates").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    assert validate_serato_root("~/Serato") == (fake_home / "Serato").resolve()


def test_read_only_root_is_rejected_before_any_write(temporary_serato_root: Path) -> None:
    with patch("serato_writer.os.access", return_value=False):
        with pytest.raises(PermissionError, match="readable and writable"):
            validate_serato_root(temporary_serato_root)


def test_real_and_configured_production_roots_are_hard_blocked(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="must never access"):
        Path(REAL_SERATO_ROOT).exists()

    configured = tmp_path / "configured-production-root"
    monkeypatch.setenv("SERATO_PRODUCTION_ROOT", str(configured))
    with pytest.raises(AssertionError, match="must never access"):
        configured.exists()


def test_backup_failure_blocks_live_write_and_result_keeps_original_bytes(
    temporary_serato_root: Path, temporary_audio_file: Path,
) -> None:
    database = temporary_serato_root / "database V2"
    before = database.read_bytes()
    with patch("serato_writer.backup_serato_folder", side_effect=OSError("backup disk full")):
        result = write_tracks_to_crates(
            temporary_serato_root, [("Backup Failure", temporary_audio_file)], dry_run=False
        )[0]
    assert (result.success, result.changed, result.status) == (False, False, "failed")
    assert "Backup failed" in result.error
    assert result.backup_path == ""
    assert database.read_bytes() == before
    assert not (temporary_serato_root / "SubCrates" / "Backup Failure.crate").exists()


def test_live_results_include_backup_path_and_atomic_failures_preserve_full_error(
    temporary_serato_root: Path, temporary_audio_file: Path,
) -> None:
    with patch("serato_writer._atomic_replace", side_effect=PermissionError("denied replacement")):
        result = write_tracks_to_crates(
            temporary_serato_root, [("Atomic Failure", temporary_audio_file)], dry_run=False
        )[0]
    assert not result.success
    assert result.status == "failed"
    assert "PermissionError: denied replacement" in result.error
    assert result.backup_path
    assert Path(result.backup_path).is_dir()
    assert not (temporary_serato_root / "SubCrates" / "Atomic Failure.crate").exists()


@pytest.mark.parametrize("failure_target", ["temporary-file", "validation", "replacement"])
def test_atomic_write_failure_modes_never_replace_existing_crate(
    temporary_serato_root: Path, temporary_audio_file: Path, valid_crate_file: Path, failure_target: str,
) -> None:
    before = valid_crate_file.read_bytes()
    addition = temporary_audio_file.parent / "atomic addition.mp3"
    addition.write_bytes(b"separate disposable track")
    if failure_target == "temporary-file":
        target = "serato_writer.tempfile.mkstemp"
        side_effect = OSError("temporary file unavailable")
    elif failure_target == "validation":
        target = "serato_writer.parse_serato_crate"
        side_effect = SeratoCrateParseError(valid_crate_file, 0, 0, "temporary validation rejected")
    else:
        target = "serato_writer.os.replace"
        side_effect = PermissionError("replacement denied")

    with patch(target, side_effect=side_effect):
        result = write_tracks_to_crates(
            temporary_serato_root, [("Valid", addition)], dry_run=False, make_backup=False
        )[0]

    assert (result.success, result.changed, result.status) == (False, False, "failed")
    assert str(side_effect) in result.error
    assert valid_crate_file.read_bytes() == before


@pytest.mark.parametrize("status", ["added", "already_present", "dry_run", "failed"])
def test_write_result_contract_exposes_every_report_field(status: str) -> None:
    result = SeratoWriteResult("Raw%%Crate", "/tmp/track.mp3", status != "failed", status == "added", status)
    assert result.crate_name == "Raw%%Crate"
    assert result.track_path == "/tmp/track.mp3"
    assert isinstance(result.success, bool)
    assert isinstance(result.changed, bool)
    assert result.status == status
    assert isinstance(result.error, str)
    assert isinstance(result.backup_path, str)
