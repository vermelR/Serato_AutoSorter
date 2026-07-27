from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from approval_service import apply_approved_rows
from serato_writer import write_tracks_to_crates
from tag_writer import TagWriteResult, write_genre_year
from watcher import append_pending, load_pending_queue


pytestmark = pytest.mark.unit


@dataclass
class CrateResult:
    crate_name: str
    track_path: str
    success: bool
    changed: bool
    status: str
    error: str = ""
    backup_path: str = ""


def _approved_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": "/tmp/one.mp3",
                "Genre": "House",
                "Year": "2024",
                "Final Crates": ["House%%Club", "House%%Deep"],
            },
            {
                "path": "/tmp/one.mp3",
                "Genre": "must not tag twice",
                "Year": "1999",
                "Final Crates": ["House%%Deep"],
            },
        ]
    )


def test_tag_writer_noop_does_not_open_audio() -> None:
    with patch("tag_writer.MutagenFile") as mutagen_file:
        result = write_genre_year("/tmp/song.mp3")
    assert result == TagWriteResult("/tmp/song.mp3", True, "nothing to write")
    mutagen_file.assert_not_called()


def test_tag_writer_writes_genre_and_year_once_to_disposable_mock() -> None:
    audio = MagicMock()
    audio.tags = {}
    with patch("tag_writer.MutagenFile", return_value=audio):
        result = write_genre_year("/tmp/song.mp3", genre="House", year="2024")
    assert result == TagWriteResult("/tmp/song.mp3", True)
    assert audio.tags == {"genre": ["House"], "date": ["2024"]}
    audio.save.assert_called_once_with()


def test_tag_writer_missing_or_unsupported_file_has_clear_failure() -> None:
    with patch("tag_writer.MutagenFile", side_effect=FileNotFoundError("missing file")):
        missing = write_genre_year("/tmp/missing.mp3", genre="House")
    with patch("tag_writer.MutagenFile", return_value=None):
        unsupported = write_genre_year("/tmp/unsupported.bin", genre="House")
    assert not missing.success and "FileNotFoundError: missing file" in missing.error
    assert not unsupported.success and "Unsupported/unrecognized" in unsupported.error


def test_multiple_final_crates_tag_once_and_write_every_raw_assignment() -> None:
    tag_writer = MagicMock(return_value=TagWriteResult("/tmp/one.mp3", True))
    crate_writer = MagicMock(
        return_value=[
            CrateResult("House%%Club", "/tmp/one.mp3", True, True, "added"),
            CrateResult("House%%Deep", "/tmp/one.mp3", True, True, "added"),
        ]
    )
    tags, crates = apply_approved_rows(
        _approved_frame(),
        "/temporary/serato",
        dry_run=False,
        make_backup=True,
        tag_writer=tag_writer,
        crate_writer=crate_writer,
    )
    tag_writer.assert_called_once_with("/tmp/one.mp3", genre="House", year="2024")
    assert crate_writer.call_args.kwargs["assignments"] == [
        ("House%%Club", "/tmp/one.mp3"),
        ("House%%Deep", "/tmp/one.mp3"),
    ]
    assert tags == [{"path": "/tmp/one.mp3", "success": True, "error": ""}]
    assert [row["crate_name"] for row in crates] == ["House%%Club", "House%%Deep"]
    assert all("›" not in row["crate_name"] for row in crates)


def test_dry_run_writes_neither_tags_nor_files_and_reports_separate_results() -> None:
    tag_writer = MagicMock()
    crate_writer = MagicMock(
        return_value=[CrateResult("House%%Club", "/tmp/one.mp3", True, False, "dry_run")]
    )
    tags, crates = apply_approved_rows(
        _approved_frame().iloc[[0]],
        "/temporary/serato",
        dry_run=True,
        make_backup=True,
        tag_writer=tag_writer,
        crate_writer=crate_writer,
    )
    tag_writer.assert_not_called()
    assert tags == [{"path": "/tmp/one.mp3", "success": True, "error": "DRY_RUN"}]
    assert crates[0]["status"] == "dry_run"
    assert crate_writer.call_args.kwargs["dry_run"] is True


def test_dry_run_preserves_crates_database_audio_backups_and_queue_bytes(
    temporary_serato_root: Path, temporary_audio_file: Path,
) -> None:
    append_pending({"path": str(temporary_audio_file), "state": "pending"})
    before_root = {
        str(path.relative_to(temporary_serato_root)): path.read_bytes()
        for path in temporary_serato_root.rglob("*") if path.is_file()
    }
    before_audio = temporary_audio_file.read_bytes()
    before_queue = list(load_pending_queue())
    tags, crates = apply_approved_rows(
        pd.DataFrame([{
            "path": str(temporary_audio_file),
            "Genre": "House",
            "Year": "2024",
            "Final Crates": ["House%%Club"],
        }]),
        temporary_serato_root,
        dry_run=True,
        make_backup=True,
        tag_writer=MagicMock(),
        crate_writer=write_tracks_to_crates,
    )
    after_root = {
        str(path.relative_to(temporary_serato_root)): path.read_bytes()
        for path in temporary_serato_root.rglob("*") if path.is_file()
    }
    assert tags[0]["error"] == "DRY_RUN"
    assert crates[0]["status"] == "dry_run"
    assert after_root == before_root
    assert temporary_audio_file.read_bytes() == before_audio
    assert load_pending_queue() == before_queue
    assert not (temporary_serato_root.parent / "SeratoAI_Backups").exists()


def test_tag_and_crate_failures_remain_independently_visible() -> None:
    tag_writer = MagicMock(return_value=TagWriteResult("/tmp/one.mp3", False, "tag save failed"))
    crate_writer = MagicMock(
        return_value=[CrateResult("House%%Club", "/tmp/one.mp3", False, False, "failed", "crate denied")]
    )
    tags, crates = apply_approved_rows(
        _approved_frame().iloc[[0]],
        "/temporary/serato",
        dry_run=False,
        make_backup=False,
        tag_writer=tag_writer,
        crate_writer=crate_writer,
    )
    assert tags[0]["success"] is False and tags[0]["error"] == "tag save failed"
    assert crates[0]["success"] is False and crates[0]["error"] == "crate denied"


def test_unexpected_crate_writer_exception_is_reported_per_assignment_without_losing_tag_result() -> None:
    tag_writer = MagicMock(return_value=TagWriteResult("/tmp/one.mp3", True))
    crate_writer = MagicMock(side_effect=OSError("writer interrupted"))
    tags, crates = apply_approved_rows(
        _approved_frame().iloc[[0]],
        "/temporary/serato",
        dry_run=False,
        make_backup=False,
        tag_writer=tag_writer,
        crate_writer=crate_writer,
    )
    assert tags == [{"path": "/tmp/one.mp3", "success": True, "error": ""}]
    assert [(row["crate_name"], row["success"], row["status"]) for row in crates] == [
        ("House%%Club", False, "failed"),
        ("House%%Deep", False, "failed"),
    ]
    assert all("OSError: writer interrupted" in row["error"] for row in crates)
