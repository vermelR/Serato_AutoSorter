from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from serato_ai.core.models import CrateWriteResult

from serato_crate import (
    CrateTrackEntry,
    SeratoCrateParseError,
    canonical_crate_name,
    canonical_track_path,
    parse_serato_crate,
    parse_serato_crate_bytes,
    serato_serialized_track_path,
    serialize_serato_track_record,
)

# pyserato is used only to construct the standard empty crate header.  Its
# internal parser is deliberately not used: it decodes unvalidated UTF-16
# chunks and is the source of the former odd-byte UnicodeDecodeError.
try:
    from pyserato.builder import Builder
    from pyserato.model.crate import Crate
except Exception as exc:  # pragma: no cover - installation issue, not I/O logic
    raise ImportError(
        "Could not import pyserato's crate serializer. Install pyserato in this venv."
    ) from exc


# Compatibility name retained for integrations and regression tests. The
# canonical cross-layer contract now lives in ``serato_ai.core.models``.
SeratoWriteResult = CrateWriteResult


@dataclass(frozen=True)
class _Assignment:
    crate_name: str
    crate_key: str
    track_path: Path
    track_key: str


def backup_serato_folder(serato_root: Path, backup_dir: Optional[Path] = None) -> Path:
    serato_root = serato_root.expanduser().resolve()
    if not serato_root.is_dir():
        raise FileNotFoundError(f"Serato folder not found: {serato_root}")

    if backup_dir is None:
        backup_dir = serato_root.parent / "SeratoAI_Backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = backup_dir / f"{serato_root.name}_backup_{stamp}"
    shutil.copytree(serato_root, destination)
    return destination


def restore_serato_backup(serato_root: Path, backup_path: Path) -> Path:
    """Restore an explicit Serato-root snapshot without touching the backup.

    The backup is copied to a sibling staging directory before the live root is
    moved aside. If activation fails, the original root is moved back. This is
    intentionally a library-level operation: callers must opt in to restoring
    a particular backup, and automated tests use only temporary roots.
    """
    root = Path(serato_root).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    if not backup.is_dir():
        raise FileNotFoundError(f"Serato backup is not a directory: {backup}")
    if backup == root or root in backup.parents:
        raise ValueError("Backup must be outside the Serato root being restored")

    parent = root.parent
    token = uuid.uuid4().hex
    staging = parent / f".{root.name}.restore-staging-{token}"
    previous = parent / f".{root.name}.pre-restore-{token}"
    activated = False

    try:
        # copy2 (used by copytree's default) retains file contents and metadata
        # for normal files, while symlinks are preserved rather than followed.
        shutil.copytree(backup, staging, symlinks=True)
        if root.exists():
            os.replace(root, previous)
        os.replace(staging, root)
        activated = True
    except Exception as exc:
        # If the staging activation failed after the live root was moved,
        # restore that root before reporting the failure. Keep the explicit
        # backup untouched in all cases.
        if previous.exists() and not root.exists():
            try:
                os.replace(previous, root)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Serato backup restore failed and rollback also failed: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                ) from exc
        raise RuntimeError(
            f"Serato backup restore failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if activated and previous.exists():
            shutil.rmtree(previous)

    return root


def _subcrates_dir(serato_root: Path) -> Path:
    """Return the existing Subcrates directory, preserving its on-disk case."""
    for name in ("SubCrates", "Subcrates"):
        candidate = serato_root / name
        if candidate.is_dir():
            return candidate
    return serato_root / "SubCrates"


def validate_serato_root(serato_root: str | Path) -> Path:
    """Resolve and validate the *selected* Serato root before a live write.

    A missing ``SubCrates`` directory is almost always an incorrectly selected
    folder (or a backup folder).  Reject it instead of silently creating a
    lookalike Serato structure somewhere else.  The function is intentionally
    side-effect free so the app and tests can report a clear problem before
    any backup or crate write is attempted.
    """
    root = Path(serato_root).expanduser().resolve()
    if root.name == "SeratoAI_Backups" or "_backup_" in root.name:
        raise ValueError("Selected Serato root appears to be a backup directory")
    if not root.is_dir():
        raise FileNotFoundError(f"Serato root is not a directory: {root}")

    subcrates = _subcrates_dir(root)
    if not subcrates.is_dir():
        raise FileNotFoundError(
            f"Serato root is missing its SubCrates directory: {root}"
        )

    # Access checks provide an early readable error.  _atomic_replace still
    # handles races and platform-specific permission failures safely.
    required_access = os.R_OK | os.W_OK | os.X_OK
    if not os.access(root, required_access) or not os.access(subcrates, required_access):
        raise PermissionError(f"Serato root is not readable and writable: {root}")
    return root


def _crate_file_path(serato_root: Path, crate_name: str) -> Path:
    if not crate_name or Path(crate_name).name != crate_name:
        raise ValueError("Crate name must be a filename, not a path")
    subcrates = _subcrates_dir(serato_root)
    return subcrates / f"{crate_name}.crate"


def _result(
    assignment: _Assignment,
    *,
    success: bool,
    changed: bool,
    status: str,
    error: str = "",
    backup_path: str = "",
) -> SeratoWriteResult:
    return SeratoWriteResult(
        crate_name=assignment.crate_name,
        track_path=str(assignment.track_path),
        success=success,
        changed=changed,
        status=status,
        error=error,
        backup_path=backup_path,
    )


def _new_crate_bytes() -> bytes:
    """Build pyserato's normal binary header; never create an empty text file."""
    crate = Crate("SeratoAI Writer")
    contents = Builder()._construct(crate)  # type: ignore[attr-defined]
    parse_serato_crate_bytes(contents, "<new Serato crate>")
    return contents


def _existing_path_keys(entries: list[CrateTrackEntry]) -> set[str]:
    keys: set[str] = set()
    for entry in entries:
        if not entry.path:
            continue
        _, key = canonical_track_path(entry.path)
        keys.add(key)
    return keys


def _atomic_replace(crate_file: Path, contents: bytes) -> None:
    """Validate a temporary binary crate then atomically replace the original."""
    crate_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{crate_file.name}.", suffix=".tmp", dir=crate_file.parent
    )
    temporary_file = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())

        # If this fails, leave the original untouched and retain the temporary
        # file for diagnosis, as it may contain useful evidence.
        parse_serato_crate(temporary_file)
        os.replace(temporary_file, crate_file)
    except Exception:
        raise


def _ordered_results(
    input_failures: list[SeratoWriteResult],
    assignments: list[_Assignment],
    outcomes: dict[tuple[str, str], SeratoWriteResult],
) -> list[SeratoWriteResult]:
    return input_failures + [
        outcomes[(assignment.crate_key, assignment.track_key)] for assignment in assignments
    ]


def write_tracks_to_crates(
    serato_root: Path,
    assignments: Iterable[tuple[str, str | Path]],
    dry_run: bool = True,
    make_backup: bool = True,
) -> list[SeratoWriteResult]:
    """Safely add unique, canonical path references to selected Serato crates.

    This operation only updates ``Subcrates/*.crate`` files.  It never reads or
    writes ``database V2`` or any other global Serato library index.
    """
    root = Path(serato_root).expanduser().resolve()
    input_failures: list[SeratoWriteResult] = []
    normalized: list[_Assignment] = []
    seen_assignments: set[tuple[str, str]] = set()

    for crate_value, track_value in assignments:
        try:
            crate_name, crate_key = canonical_crate_name(crate_value)
            if not crate_name:
                raise ValueError("Empty crate name")
            # Resolve the target early so unsafe crate names can never escape
            # the selected Serato root.
            _crate_file_path(root, crate_name)
            track_path, track_key = canonical_track_path(track_value)
            if not track_path.is_file():
                raise FileNotFoundError("Track file not found or is not a regular file")
            assignment_key = (crate_key, track_key)
            if assignment_key in seen_assignments:
                continue
            seen_assignments.add(assignment_key)
            normalized.append(_Assignment(crate_name, crate_key, track_path, track_key))
        except Exception as exc:
            raw_crate, _ = canonical_crate_name(crate_value)
            try:
                display_path = str(canonical_track_path(track_value)[0])
            except Exception:
                display_path = str(track_value)
            input_failures.append(
                SeratoWriteResult(
                    crate_name=raw_crate,
                    track_path=display_path,
                    success=False,
                    changed=False,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not normalized:
        return input_failures

    try:
        root = validate_serato_root(root)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return input_failures + [
            _result(item, success=False, changed=False, status="failed", error=error)
            for item in normalized
        ]

    grouped: dict[str, list[_Assignment]] = defaultdict(list)
    crate_names: dict[str, str] = {}
    for assignment in normalized:
        grouped[assignment.crate_key].append(assignment)
        crate_names.setdefault(assignment.crate_key, assignment.crate_name)

    outcomes: dict[tuple[str, str], SeratoWriteResult] = {}
    crate_contents: dict[str, bytes] = {}
    existing_keys: dict[str, set[str]] = {}

    # Read and validate every original crate before making a backup or writing
    # anything. A malformed crate is never rewritten.
    for crate_key, group in grouped.items():
        crate_file = _crate_file_path(root, crate_names[crate_key])
        try:
            if crate_file.exists():
                original = crate_file.read_bytes()
                entries = parse_serato_crate(crate_file)
            else:
                original = b""
                entries = []
            crate_contents[crate_key] = original
            existing_keys[crate_key] = _existing_path_keys(entries)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for assignment in group:
                outcomes[(assignment.crate_key, assignment.track_key)] = _result(
                    assignment,
                    success=False,
                    changed=False,
                    status="failed",
                    error=error,
                )

    plans: dict[str, list[_Assignment]] = {}
    for crate_key, group in grouped.items():
        if crate_key not in existing_keys:
            continue
        additions: list[_Assignment] = []
        for assignment in group:
            key = (assignment.crate_key, assignment.track_key)
            if assignment.track_key in existing_keys[crate_key]:
                outcomes[key] = _result(
                    assignment,
                    success=True,
                    changed=False,
                    status="already_present",
                )
            else:
                additions.append(assignment)
                # A grouped assignment is unique, so it also prevents two
                # records for one file when the same crate is selected twice.
                existing_keys[crate_key].add(assignment.track_key)
        plans[crate_key] = additions

    if dry_run:
        for additions in plans.values():
            for assignment in additions:
                outcomes[(assignment.crate_key, assignment.track_key)] = _result(
                    assignment,
                    success=True,
                    changed=False,
                    status="dry_run",
                )
        return _ordered_results(input_failures, normalized, outcomes)

    any_changes = any(plans.values())
    if not any_changes:
        return _ordered_results(input_failures, normalized, outcomes)

    backup_path = ""
    if make_backup:
        try:
            backup = backup_serato_folder(root)
            backup_path = str(backup)
            print(f"Serato backup created: {backup}")
        except Exception as exc:
            error = f"Backup failed: {type(exc).__name__}: {exc}"
            for additions in plans.values():
                for assignment in additions:
                    outcomes[(assignment.crate_key, assignment.track_key)] = _result(
                        assignment,
                        success=False,
                        changed=False,
                        status="failed",
                        error=error,
                        backup_path=backup_path,
                    )
            return _ordered_results(input_failures, normalized, outcomes)

    for crate_key, additions in plans.items():
        if not additions:
            continue
        crate_file = _crate_file_path(root, crate_names[crate_key])
        try:
            contents = crate_contents[crate_key] or _new_crate_bytes()
            for assignment in additions:
                contents += serialize_serato_track_record(
                    serato_serialized_track_path(assignment.track_path)
                )

            # Validate the completed bytes before they can replace a real
            # crate, then validate the on-disk temporary file in _atomic_replace.
            parsed_paths = {
                canonical_track_path(entry.path)[1]
                for entry in parse_serato_crate_bytes(contents, crate_file)
                if entry.path
            }
            missing = [item for item in additions if item.track_key not in parsed_paths]
            if missing:
                raise SeratoCrateParseError(
                    crate_file,
                    0,
                    len(contents),
                    "Serialized crate did not contain every requested track path",
                )
            _atomic_replace(crate_file, contents)
            for assignment in additions:
                outcomes[(assignment.crate_key, assignment.track_key)] = _result(
                    assignment,
                    success=True,
                    changed=True,
                    status="added",
                    backup_path=backup_path,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for assignment in additions:
                outcomes[(assignment.crate_key, assignment.track_key)] = _result(
                    assignment,
                    success=False,
                    changed=False,
                    status="failed",
                    error=error,
                    backup_path=backup_path,
                )

    return _ordered_results(input_failures, normalized, outcomes)
