from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from serato_crate import parse_serato_crate, serialize_serato_track_record
from serato_writer import restore_serato_backup, write_tracks_to_crates


pytestmark = [pytest.mark.writer, pytest.mark.integration]


def _tree_manifest(root: Path) -> dict[str, tuple[bytes, int]]:
    """Exact temporary-root snapshot: every relative file, byte payload, and mode."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def temporary_serato_environment():
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        source_audio = base / "source-audio.mp3"
        source_audio.write_bytes(b"temporary copied-audio fixture")
        copied_audio = base / "Copied Test Track.mp3"
        shutil.copy2(source_audio, copied_audio)
        existing_audio = base / "Existing Track.mp3"
        existing_audio.write_bytes(b"existing fixture")

        root = base / "_Serato_"
        subcrates = root / "SubCrates"
        subcrates.mkdir(parents=True)
        (root / "database V2").write_bytes(b"global database sentinel: must never change")
        (root / "preferences.json").write_text('{"sentinel": true}', encoding="utf-8")

        # A realistic existing crate fixture: valid header plus one raw
        # no-leading-slash Serato path reference.
        existing_crate = subcrates / "Existing.crate"
        existing_crate.write_bytes(
            b"vrsn\x00\x00\x00\x00"
            + serialize_serato_track_record(str(existing_audio.resolve()).lstrip("/"))
        )

        yield {
            "base": base,
            "root": root,
            "copied_audio": copied_audio,
            "existing_audio": existing_audio,
            "existing_crate": existing_crate,
        }


def _crate_paths(crate_file: Path) -> list[str]:
    return [entry.path for entry in parse_serato_crate(crate_file)]


def test_temporary_end_to_end_backup_dry_run_live_dedup_and_restore(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    copied_audio = env["copied_audio"]
    existing_audio = env["existing_audio"]
    existing_crate = env["existing_crate"]
    original_manifest = _tree_manifest(root)
    original_existing_bytes = existing_crate.read_bytes()
    assignments = [
        ("Existing", str(copied_audio)),
        ("New One", copied_audio),
        ("New One", str(copied_audio)),
        ("New Two", copied_audio),
    ]

    dry_results = write_tracks_to_crates(root, assignments, dry_run=True, make_backup=True)

    assert [(result.crate_name, result.success, result.changed, result.status) for result in dry_results] == [
        ("Existing", True, False, "dry_run"),
        ("New One", True, False, "dry_run"),
        ("New Two", True, False, "dry_run"),
    ]
    assert _tree_manifest(root) == original_manifest
    assert not (env["base"] / "SeratoAI_Backups").exists()

    live_results = write_tracks_to_crates(root, assignments, dry_run=False, make_backup=True)

    assert [(result.success, result.changed, result.status, result.error) for result in live_results] == [
        (True, True, "added", ""),
        (True, True, "added", ""),
        (True, True, "added", ""),
    ]
    copied_path = str(copied_audio.resolve())
    assert _crate_paths(existing_crate) == [str(existing_audio.resolve()), copied_path]
    assert existing_crate.read_bytes().startswith(original_existing_bytes)
    assert _crate_paths(root / "SubCrates" / "New One.crate") == [copied_path]
    assert _crate_paths(root / "SubCrates" / "New Two.crate") == [copied_path]
    assert (root / "database V2").read_bytes() == original_manifest["database V2"][0]

    backups = list((env["base"] / "SeratoAI_Backups").glob("_Serato__backup_*"))
    assert len(backups) == 1
    backup = backups[0]
    assert _tree_manifest(backup) == original_manifest

    repeated = write_tracks_to_crates(root, assignments, dry_run=False, make_backup=True)
    assert [(result.success, result.changed, result.status) for result in repeated] == [
        (True, False, "already_present"),
        (True, False, "already_present"),
        (True, False, "already_present"),
    ]
    assert len(_crate_paths(existing_crate)) == 2
    assert len(_crate_paths(root / "SubCrates" / "New One.crate")) == 1
    assert len(_crate_paths(root / "SubCrates" / "New Two.crate")) == 1

    restore_serato_backup(root, backup)
    assert _tree_manifest(root) == original_manifest
    assert backup.is_dir()


def test_malformed_crate_produces_a_failed_result_without_file_corruption(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    broken = root / "SubCrates" / "Broken.crate"
    original_bytes = b"vrsn\x00\x00\x00\x00otrk\x00\x00\x00\x09ptrk\x00\x00\x00\x01/"
    broken.write_bytes(original_bytes)
    before = _tree_manifest(root)

    result = write_tracks_to_crates(
        root, [("Broken", env["copied_audio"])], dry_run=False, make_backup=True
    )

    assert len(result) == 1
    assert (result[0].success, result[0].changed, result[0].status) == (False, False, "failed")
    assert "expected an even UTF-16 byte length" in result[0].error
    assert broken.read_bytes() == original_bytes
    assert _tree_manifest(root) == before
    assert not (env["base"] / "SeratoAI_Backups").exists()


def test_atomic_failure_reports_only_that_assignment_and_preserves_original_crate(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    existing_crate = env["existing_crate"]
    original = existing_crate.read_bytes()

    with patch("serato_writer._atomic_replace", side_effect=OSError("simulated replace failure")):
        results = write_tracks_to_crates(
            root, [("Existing", env["copied_audio"])], dry_run=False, make_backup=False
        )

    assert len(results) == 1
    assert (results[0].success, results[0].changed, results[0].status) == (False, False, "failed")
    assert "simulated replace failure" in results[0].error
    assert existing_crate.read_bytes() == original


def test_partial_results_keep_good_crates_and_reject_malformed_ones(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    broken = root / "SubCrates" / "Broken.crate"
    broken.write_bytes(b"")

    results = write_tracks_to_crates(
        root,
        [("Good", env["copied_audio"]), ("Broken", env["copied_audio"])],
        dry_run=False,
        make_backup=True,
    )

    assert [(result.crate_name, result.success, result.status) for result in results] == [
        ("Good", True, "added"),
        ("Broken", False, "failed"),
    ]
    assert _crate_paths(root / "SubCrates" / "Good.crate") == [str(env["copied_audio"].resolve())]
    assert broken.read_bytes() == b""
    assert "Serato crate file is empty" in results[1].error


def test_restore_after_multiple_live_writes_recreates_the_original_snapshot(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    original = _tree_manifest(root)
    first = write_tracks_to_crates(
        root, [("First", env["copied_audio"])], dry_run=False, make_backup=True
    )[0]
    backup = Path(first.backup_path)
    second = write_tracks_to_crates(
        root, [("Second", env["copied_audio"])], dry_run=False, make_backup=False
    )[0]
    assert first.success and second.success
    assert (root / "SubCrates" / "First.crate").exists()
    assert (root / "SubCrates" / "Second.crate").exists()

    restore_serato_backup(root, backup)
    assert _tree_manifest(root) == original
    assert backup.is_dir()


def test_restore_activation_failure_keeps_live_root_and_backup_intact(
    temporary_serato_environment,
) -> None:
    env = temporary_serato_environment
    root = env["root"]
    original = _tree_manifest(root)
    backup = env["base"] / "restore-source"
    shutil.copytree(root, backup)
    real_replace = __import__("os").replace

    def fail_staging_activation(source, destination):
        if (
            Path(source).name.startswith(f".{root.name}.restore-staging-")
            and Path(destination).resolve() == root.resolve()
        ):
            raise OSError("simulated activation failure")
        return real_replace(source, destination)

    with patch("serato_writer.os.replace", side_effect=fail_staging_activation):
        with pytest.raises(RuntimeError, match="Serato backup restore failed: OSError: simulated activation failure"):
            restore_serato_backup(root, backup)

    assert _tree_manifest(root) == original
    assert backup.is_dir()
