"""
Background watcher: keeps an eye on your "new music" folders (Downloads,
an inbox folder, etc.) and/or a Serato "New"/"Unsorted" inbox crate, and
automatically runs crate + genre + year predictions on anything new it
finds. Results are queued to a small JSONL file that the Streamlit app polls
and shows in the review table — nothing gets written into Serato or into
your files until you hit Approve + Apply there.

Runs as a simple polling loop in a background thread (no OS-level file
watcher dependency needed). A file is only processed once its size has
stopped changing for FILE_STABLE_SECONDS, so partially-downloaded files
aren't analyzed.
"""

from __future__ import annotations

import threading
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import config
from phase1_harvest import parse_serato_crate
from phase4_engine import AUDIO_EXTS, load_model_bundle, normalize_path, propose_crates_for_files
from serato_ai.infrastructure.queue_store import QueueStore

if TYPE_CHECKING:
    from serato_ai.services.prediction_service import PredictionService


def _queue_store() -> QueueStore:
    """Build a store at call time so config overrides remain testable."""
    return QueueStore(Path(config.PENDING_QUEUE_PATH), Path(config.PROCESSED_INDEX_PATH))


def load_processed_index() -> set[str]:
    return _queue_store().load_processed()


def save_processed_index(index: set[str]) -> None:
    _queue_store().save_processed(index)


def load_pending_queue(
    *,
    limit: int | None = None,
    offset: int = 0,
    search: str = "",
    view: str = "All",
    filter_signatures: tuple[str, ...] | None = None,
) -> list[dict]:
    return _queue_store().load_pending(
        limit=limit,
        offset=offset,
        search=search,
        view=view,
        filter_signatures=filter_signatures,
    )


def count_pending_queue(
    *,
    search: str = "",
    view: str = "All",
    filter_signatures: tuple[str, ...] | None = None,
) -> int:
    return _queue_store().count_pending(
        search=search,
        view=view,
        filter_signatures=filter_signatures,
    )


def append_pending(row: dict) -> bool:
    return _queue_store().append_pending(row)


def append_pending_many(rows: list[dict]) -> tuple[bool, ...]:
    return _queue_store().append_many(rows)


def update_pending_queue(rows: list[dict]) -> int:
    return _queue_store().update_pending(rows)


def relocate_pending(file_identity: str, new_path: str) -> str:
    return _queue_store().relocate_pending(file_identity, new_path)


def mark_reviewed(paths: set[str]) -> None:
    """Remove tracks from the pending queue and remember them as processed,
    so they aren't re-queued on the watcher's next scan. Safe to call even
    if no Watcher is currently running (e.g. reviewing a previous session's
    queue)."""
    _queue_store().mark_reviewed(paths)


def remove_pending(paths: set[str]) -> None:
    """Drop entries (by 'path') from the pending queue, e.g. after they've
    been approved/denied in the app."""
    _queue_store().remove_pending(paths)


@dataclass
class _StableTracker:
    size: int
    modified_ns: int
    since: float
    samples: int = 1


class Watcher:
    def __init__(
        self,
        folders: list[str],
        crates: list[str] | None = None,
        model_path: str = config.CRATE_MODEL_PATH,
        poll_seconds: float = 2.0,
        topk: int = 3,
        identify_genre: bool = True,
        excluded_crates: set[str] | None = None,
        allowed_crates: set[str] | None = None,
        prediction_service: "PredictionService | None" = None,
    ):
        self.folders = folders
        self.crates = crates or []
        self.model_path = model_path
        self.poll_seconds = poll_seconds
        self.topk = topk
        self.identify_genre = identify_genre
        self.excluded_crates = excluded_crates or set()
        self.allowed_crates = allowed_crates
        self.prediction_service = prediction_service

        self._processed = load_processed_index()
        self._stable: dict[str, _StableTracker] = {}
        pending = load_pending_queue()
        self._pending_paths = {str(r.get("path")) for r in pending if r.get("path")}
        self._pending_identities = {
            str(r.get("_file_identity")) for r in pending if r.get("_file_identity")
        }
        self._crate_cache: dict[str, tuple[tuple[int, int], tuple[str, ...]]] = {}
        self._failure_retry_after: dict[str, float] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_error: str = ""
        self._last_scan_ts: float = 0.0
        self._bundle = None

    # -------------------------
    # lifecycle
    # -------------------------
    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "last_scan": self._last_scan_ts,
            "last_error": self._last_error,
            "queued": count_pending_queue(),
        }

    def mark_processed(self, paths: set[str]) -> None:
        """Call after tracks are approved/applied (or denied) in the app so
        they aren't re-queued on the next scan."""
        self._processed |= paths
        self._pending_paths -= paths
        for row in load_pending_queue():
            if row.get("path") in paths and row.get("_file_identity"):
                self._processed.add(str(row["_file_identity"]))
                self._pending_identities.discard(str(row["_file_identity"]))
        save_processed_index(self._processed)
        remove_pending(paths)

    def set_crate_filters(
        self,
        *,
        allowed_crates: set[str] | None,
        excluded_crates: set[str] | None,
    ) -> None:
        """Apply sidebar category/exclusion changes on the next watcher scan."""
        self.allowed_crates = allowed_crates
        self.excluded_crates = excluded_crates or set()

    # -------------------------
    # internals
    # -------------------------
    def _run(self) -> None:
        try:
            if self.prediction_service is None:
                self._bundle = load_model_bundle(self.model_path)
            else:
                self._bundle = self.prediction_service.load_bundle(self.model_path)
        except Exception as e:
            self._last_error = f"Could not load crate model: {e}"
            return

        while not self._stop_event.is_set():
            try:
                self._scan_once()
                self._last_scan_ts = time.time()
                self._last_error = ""
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
            self._stop_event.wait(self.poll_seconds)

    def _candidate_files(self) -> list[Path]:
        found: dict[str, Path] = {}

        for folder in self.folders:
            p = Path(normalize_path(folder))
            if not p.is_dir():
                continue
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    found[str(f.resolve())] = f

        for crate_path in self.crates:
            cp = Path(normalize_path(crate_path))
            if not cp.is_file():
                continue
            try:
                stat = cp.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                continue
            cached = self._crate_cache.get(str(cp))
            if cached is not None and cached[0] == signature:
                crate_paths = cached[1]
            else:
                try:
                    crate_paths = tuple(parse_serato_crate(str(cp)))
                    self._crate_cache[str(cp)] = (signature, crate_paths)
                except Exception:
                    # A rapid partial crate update retains the last valid view.
                    crate_paths = cached[1] if cached is not None else ()
            for track_path in crate_paths:
                f = Path(normalize_path(track_path))
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    found[str(f.resolve())] = f

        return [found[key] for key in sorted(found)]

    @staticmethod
    def _file_identity(f: Path) -> str:
        stat = f.stat()
        if int(getattr(stat, "st_ino", 0)):
            return f"inode:{int(getattr(stat, 'st_dev', 0))}:{int(stat.st_ino)}"
        return f"path:{str(f.resolve())}:{stat.st_size}:{stat.st_mtime_ns}"

    def _is_stable(self, f: Path) -> bool:
        key = str(f)
        try:
            stat = f.stat()
            size = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            return False
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS or not os.access(f, os.R_OK):
            return False
        lowered = f.name.casefold()
        if lowered.startswith(".") or lowered.endswith((".part", ".download", ".crdownload", ".tmp")):
            return False

        now = time.time()
        tracker = self._stable.get(key)

        if tracker is None or tracker.size != size or tracker.modified_ns != modified_ns:
            self._stable[key] = _StableTracker(size=size, modified_ns=modified_ns, since=now)
            return False

        self._stable[key] = _StableTracker(
            size=size,
            modified_ns=modified_ns,
            since=tracker.since,
            samples=tracker.samples + 1,
        )
        return tracker.samples >= 1 and (now - tracker.since) >= config.FILE_STABLE_SECONDS

    def _scan_once(self) -> None:
        # Keep candidates pending while the sidebar leaves no crate available;
        # they can be predicted after the user relaxes an exclusion instead of
        # being incorrectly marked processed as failures.
        if self.allowed_crates is not None and not self.allowed_crates:
            return

        new_candidates = []
        for f in self._candidate_files():
            key = str(f)
            if not self._is_stable(f):
                continue
            try:
                identity = self._file_identity(f)
            except OSError:
                identity = f"path:{key}"
            if (
                key in self._processed
                or identity in self._processed
            ):
                continue
            if identity in self._pending_identities:
                if key not in self._pending_paths:
                    old_path = relocate_pending(identity, key)
                    if old_path:
                        self._pending_paths.discard(old_path)
                        self._pending_paths.add(key)
                continue
            if (
                key in self._pending_paths
                or time.time() < self._failure_retry_after.get(key, 0.0)
            ):
                continue
            new_candidates.append(f)

        if not new_candidates:
            return

        if self.prediction_service is None:
            pred_df, fails_df = propose_crates_for_files(
                self._bundle, new_candidates, topk=self.topk,
                identify_genre=self.identify_genre,
                excluded_crates=self.excluded_crates,
                allowed_crates=self.allowed_crates,
            )
        else:
            response = self.prediction_service.predict_files(
                self._bundle,
                new_candidates,
                topk=self.topk,
                identify_genre=self.identify_genre,
                excluded_crates=self.excluded_crates,
                allowed_crates=self.allowed_crates,
            )
            pred_df, fails_df = self.prediction_service.to_dataframes(response)

        queued_rows: list[dict] = []
        for _, row in pred_df.iterrows():
            row_dict = row.to_dict()
            try:
                row_dict["_file_identity"] = self._file_identity(Path(str(row_dict.get("path", ""))))
            except OSError:
                row_dict["_file_identity"] = ""
            queued_rows.append(row_dict)
        accepted_rows = (
            (append_pending(queued_rows[0]),)
            if len(queued_rows) == 1
            else append_pending_many(queued_rows)
        )
        for row_dict, accepted in zip(queued_rows, accepted_rows):
            if accepted:
                self._pending_paths.add(str(row_dict.get("path")))
                if row_dict["_file_identity"]:
                    self._pending_identities.add(row_dict["_file_identity"])

        # Transient read/drive failures are retried after a bounded backoff;
        # they are never silently marked processed forever.
        for _, row in fails_df.iterrows():
            path = row.get("path")
            if path:
                error = str(row.get("error", "")).casefold()
                transient = any(
                    marker in error
                    for marker in (
                        "disappeared", "unavailable", "permission", "oserror",
                        "not found", "temporar", "resource busy",
                    )
                )
                if transient:
                    self._failure_retry_after[str(path)] = time.time() + max(10.0, self.poll_seconds * 5)
                else:
                    self._processed.add(str(path))
        if self._processed:
            save_processed_index(self._processed)
