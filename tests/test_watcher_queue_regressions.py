from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from watcher import (
    Watcher,
    append_pending,
    load_pending_queue,
    load_processed_index,
    mark_reviewed,
    remove_pending,
)


pytestmark = [pytest.mark.unit, pytest.mark.watcher]


def _queue_row(path: str = "/tmp/queued.mp3") -> dict:
    return {
        "path": path,
        "Song Title": "Queued",
        "Suggested Crate": "House%%Club",
        "Confidence": 0.5,
        "_top1_crate": "House%%Club",
        "_top1_prob": 0.5,
    }


def test_empty_queue_refresh_and_remove_are_safe() -> None:
    assert load_pending_queue() == []
    remove_pending({"/tmp/not-present.mp3"})
    assert load_pending_queue() == []


def test_pending_queue_round_trip_and_reviewed_tracks_are_removed() -> None:
    append_pending(_queue_row("/tmp/one.mp3"))
    append_pending(_queue_row("/tmp/two.mp3"))
    assert [row["path"] for row in load_pending_queue()] == ["/tmp/one.mp3", "/tmp/two.mp3"]

    mark_reviewed({"/tmp/one.mp3"})
    assert [row["path"] for row in load_pending_queue()] == ["/tmp/two.mp3"]
    assert load_processed_index() == {"/tmp/one.mp3"}


def test_watcher_scans_one_and_multiple_stable_tracks_into_temporary_queue(tmp_path: Path) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    one = watched / "one.mp3"
    two = watched / "two.mp3"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    watcher = Watcher(folders=[str(watched)], allowed_crates={"House%%Club"})
    watcher._bundle = {"model": object()}
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]
    predicted = pd.DataFrame([_queue_row(str(one)), _queue_row(str(two))])

    with patch("watcher.propose_crates_for_files", return_value=(predicted, pd.DataFrame())):
        watcher._scan_once()
    assert [row["path"] for row in load_pending_queue()] == [str(one), str(two)]


def test_watcher_waits_for_supported_readable_stable_non_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "FILE_STABLE_SECONDS", 0)
    track = tmp_path / "growing.mp3"
    track.write_bytes(b"first")
    hidden = tmp_path / ".partial.mp3"
    hidden.write_bytes(b"partial")
    watcher = Watcher(folders=[str(tmp_path)])

    assert not watcher._is_stable(track)
    track.write_bytes(b"first-and-growing")
    assert not watcher._is_stable(track)
    assert watcher._is_stable(track)
    assert not watcher._is_stable(hidden)


def test_rename_updates_pending_path_by_inode_without_duplicate_prediction(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    original = watched / "original.mp3"
    original.write_bytes(b"fixture")
    watcher = Watcher(
        folders=[str(watched)],
        allowed_crates={"House%%Club"},
    )
    watcher._bundle = {"model": object()}
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]
    predicted = pd.DataFrame([_queue_row(str(original))])

    with patch(
        "watcher.propose_crates_for_files",
        return_value=(predicted, pd.DataFrame()),
    ):
        watcher._scan_once()
    queued_identity = load_pending_queue()[0]["_file_identity"]

    renamed = watched / "renamed.mp3"
    original.rename(renamed)
    with patch("watcher.propose_crates_for_files") as propose:
        watcher._scan_once()

    propose.assert_not_called()
    rows = load_pending_queue()
    assert len(rows) == 1
    assert rows[0]["path"] == str(renamed)
    assert rows[0]["_file_identity"] == queued_identity


def test_watcher_does_not_enqueue_duplicate_pending_paths(tmp_path: Path) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    track = watched / "one.mp3"
    track.write_bytes(b"one")
    append_pending(_queue_row(str(track)))
    watcher = Watcher(folders=[str(watched)], allowed_crates={"House%%Club"})
    watcher._bundle = {"model": object()}
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]
    with patch("watcher.propose_crates_for_files") as propose:
        watcher._scan_once()
    propose.assert_not_called()
    assert len(load_pending_queue()) == 1


def test_prediction_failure_is_marked_processed_without_creating_a_queue_row(tmp_path: Path) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    track = watched / "failure.mp3"
    track.write_bytes(b"failure")
    watcher = Watcher(folders=[str(watched)], allowed_crates={"House%%Club"})
    watcher._bundle = {"model": object()}
    watcher._is_stable = lambda _file: True  # type: ignore[method-assign]
    failures = pd.DataFrame([{"path": str(track), "error": "features failed"}])
    with patch("watcher.propose_crates_for_files", return_value=(pd.DataFrame(), failures)):
        watcher._scan_once()
    assert str(track) in load_processed_index()
    assert load_pending_queue() == []


def test_watcher_empty_allow_list_preserves_candidates_for_later_review(tmp_path: Path) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    track = watched / "wait.mp3"
    track.write_bytes(b"wait")
    watcher = Watcher(folders=[str(watched)], allowed_crates=set())
    watcher._bundle = {"model": object()}
    watcher._candidate_files = lambda: [track]  # type: ignore[method-assign]
    with patch("watcher.propose_crates_for_files") as propose:
        watcher._scan_once()
    propose.assert_not_called()
    assert str(track) not in watcher._processed
    assert load_pending_queue() == []


def test_watcher_filter_changes_are_shared_with_manual_prediction_contract() -> None:
    watcher = Watcher(folders=[], allowed_crates={"House%%Club"}, excluded_crates={"House%%Deep"})
    watcher.set_crate_filters(
        allowed_crates={"Hip Hop%%Open Format"}, excluded_crates={"House%%Club"}
    )
    assert watcher.allowed_crates == {"Hip Hop%%Open Format"}
    assert watcher.excluded_crates == {"House%%Club"}


def test_watcher_queue_json_is_serialized_without_external_network_calls() -> None:
    append_pending({"path": "/tmp/value.mp3", "probability": np.float64(0.5)})
    stored = load_pending_queue()[0]
    assert stored == {"path": "/tmp/value.mp3", "probability": 0.5}
    assert Path(config.PENDING_QUEUE_PATH).is_file()
