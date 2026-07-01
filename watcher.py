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

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from phase1_harvest import parse_serato_crate
from phase4_engine import AUDIO_EXTS, load_model_bundle, normalize_path, propose_crates_for_files


def _load_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: str, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_processed_index() -> set[str]:
    return set(_load_json(config.PROCESSED_INDEX_PATH, []))


def save_processed_index(index: set[str]) -> None:
    _save_json(config.PROCESSED_INDEX_PATH, sorted(index))


def load_pending_queue() -> list[dict]:
    p = Path(config.PENDING_QUEUE_PATH)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_default(o):
    # Handles numpy scalar types (float64/int64/bool_) that leak in from
    # pandas rows and aren't natively JSON-serializable.
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def append_pending(row: dict) -> None:
    with open(config.PENDING_QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def mark_reviewed(paths: set[str]) -> None:
    """Remove tracks from the pending queue and remember them as processed,
    so they aren't re-queued on the watcher's next scan. Safe to call even
    if no Watcher is currently running (e.g. reviewing a previous session's
    queue)."""
    if not paths:
        return
    processed = load_processed_index()
    processed |= paths
    save_processed_index(processed)
    remove_pending(paths)


def remove_pending(paths: set[str]) -> None:
    """Drop entries (by 'path') from the pending queue, e.g. after they've
    been approved/denied in the app."""
    remaining = [r for r in load_pending_queue() if r.get("path") not in paths]
    p = Path(config.PENDING_QUEUE_PATH)
    with p.open("w", encoding="utf-8") as f:
        for r in remaining:
            f.write(json.dumps(r) + "\n")


@dataclass
class _StableTracker:
    size: int
    since: float


class Watcher:
    def __init__(
        self,
        folders: list[str],
        crates: list[str] | None = None,
        model_path: str = config.CRATE_MODEL_PATH,
        poll_seconds: float = 2.0,
        topk: int = 3,
        identify_genre: bool = True,
    ):
        self.folders = folders
        self.crates = crates or []
        self.model_path = model_path
        self.poll_seconds = poll_seconds
        self.topk = topk
        self.identify_genre = identify_genre

        self._processed = load_processed_index()
        self._stable: dict[str, _StableTracker] = {}
        self._pending_paths = {r.get("path") for r in load_pending_queue()}

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
            "queued": len(load_pending_queue()),
        }

    def mark_processed(self, paths: set[str]) -> None:
        """Call after tracks are approved/applied (or denied) in the app so
        they aren't re-queued on the next scan."""
        self._processed |= paths
        self._pending_paths -= paths
        save_processed_index(self._processed)
        remove_pending(paths)

    # -------------------------
    # internals
    # -------------------------
    def _run(self) -> None:
        try:
            self._bundle = load_model_bundle(self.model_path)
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
        found: list[Path] = []

        for folder in self.folders:
            p = Path(normalize_path(folder))
            if not p.is_dir():
                continue
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    found.append(f)

        for crate_path in self.crates:
            cp = Path(normalize_path(crate_path))
            if not cp.exists():
                continue
            for track_path in parse_serato_crate(str(cp)):
                f = Path(normalize_path(track_path))
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    found.append(f)

        return found

    def _is_stable(self, f: Path) -> bool:
        key = str(f)
        try:
            size = f.stat().st_size
        except OSError:
            return False

        now = time.time()
        tracker = self._stable.get(key)

        if tracker is None or tracker.size != size:
            self._stable[key] = _StableTracker(size=size, since=now)
            return False

        return (now - tracker.since) >= config.FILE_STABLE_SECONDS

    def _scan_once(self) -> None:
        new_candidates = []
        for f in self._candidate_files():
            key = str(f)
            if key in self._processed or key in self._pending_paths:
                continue
            if not self._is_stable(f):
                continue
            new_candidates.append(f)

        if not new_candidates:
            return

        pred_df, fails_df = propose_crates_for_files(
            self._bundle, new_candidates, topk=self.topk, identify_genre=self.identify_genre
        )

        for _, row in pred_df.iterrows():
            row_dict = row.to_dict()
            append_pending(row_dict)
            self._pending_paths.add(row_dict.get("path"))

        # Don't keep retrying files that failed feature extraction forever;
        # remember them as processed (with the error) so they don't loop.
        for _, row in fails_df.iterrows():
            path = row.get("path")
            if path:
                self._processed.add(path)

        if not fails_df.empty:
            save_processed_index(self._processed)
