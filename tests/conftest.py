"""Temporary-only fixtures and hard safety rails for the SeratoAI suite.

Nothing in this file assumes a user's home directory. The autouse guard
rejects every common filesystem operation aimed at the known production
Serato path (or an explicitly configured production root) before it can read
or mutate the library.
"""

from __future__ import annotations

import builtins
import os
import shutil
import socket
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest

import config
from serato_crate import serialize_serato_track_record


REAL_SERATO_ROOT = "/Users/diora/Music/_Serato_"


class FixturePredictionModel:
    """Pickle-safe deterministic model used by model-loading fixtures."""

    classes_ = np.array(["House%%Club", "Hip Hop%%Open Format", "House%%Deep"])

    def predict_proba(self, _features):
        return np.array([[0.5, 0.3, 0.2]])


def _path_text(value: object) -> str:
    try:
        return os.path.abspath(os.path.expanduser(os.fspath(value)))
    except (TypeError, ValueError):
        return str(value)


def _production_roots() -> tuple[str, ...]:
    configured = os.environ.get("SERATO_PRODUCTION_ROOT", "").strip()
    roots = [REAL_SERATO_ROOT]
    if configured:
        roots.append(configured)
    return tuple(_path_text(root) for root in roots)


def assert_not_production_serato_path(value: object) -> None:
    """Raise before filesystem access to the real/configured library."""
    candidate = _path_text(value)
    for root in _production_roots():
        if candidate == root or candidate.startswith(root + os.sep):
            raise AssertionError(
                "Tests must never access the production Serato library: " + candidate
            )


def _guard_one_path(original):
    def guarded(path, *args, **kwargs):
        assert_not_production_serato_path(path)
        return original(path, *args, **kwargs)

    return guarded


def _guard_path_method(original):
    def guarded(self, *args, **kwargs):
        assert_not_production_serato_path(self)
        return original(self, *args, **kwargs)

    return guarded


@pytest.fixture(autouse=True)
def isolate_filesystem_network_and_runtime(tmp_path, monkeypatch):
    """Route runtime state to ``tmp_path`` and block unsafe I/O/network calls."""
    runtime = tmp_path / "runtime-state"
    runtime.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "test-home"))
    monkeypatch.setenv("SERATO_AI_DATA_DIR", str(runtime / "seratoai-data"))
    monkeypatch.setenv("SERATO_CRATE_MODEL", str(runtime / "no-legacy-model.pkl"))
    monkeypatch.setattr(config, "PENDING_QUEUE_PATH", str(runtime / "pending.jsonl"))
    monkeypatch.setattr(config, "PROCESSED_INDEX_PATH", str(runtime / "processed.json"))
    monkeypatch.setattr(config, "DEFAULT_WATCH_FOLDERS", [str(tmp_path / "watch")])

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if not isinstance(file, int):
            assert_not_production_serato_path(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", _guard_one_path(os.open))
    for name in (
        "stat", "lstat", "listdir", "scandir", "mkdir", "makedirs", "remove", "unlink", "rmdir",
        "rename", "access", "chmod", "walk",
    ):
        monkeypatch.setattr(os, name, _guard_one_path(getattr(os, name)))

    original_replace = os.replace

    def guarded_replace(src, dst, *args, **kwargs):
        assert_not_production_serato_path(src)
        assert_not_production_serato_path(dst)
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", guarded_replace)

    for name in (
        "open", "read_bytes", "write_bytes", "read_text", "write_text", "exists", "is_file",
        "is_dir", "iterdir", "glob", "rglob", "stat", "mkdir", "unlink", "rmdir", "touch", "chmod",
    ):
        monkeypatch.setattr(Path, name, _guard_path_method(getattr(Path, name)))

    for name in ("copytree", "copy2", "copy", "copyfile", "move", "rmtree"):
        original = getattr(shutil, name)

        def guarded_shutil(src, dst=None, *args, __original=original, **kwargs):
            assert_not_production_serato_path(src)
            if dst is not None:
                assert_not_production_serato_path(dst)
            return __original(src, dst, *args, **kwargs)

        monkeypatch.setattr(shutil, name, guarded_shutil)

    def blocked_network(*_args, **_kwargs):
        raise AssertionError("Real network access is disabled during automated tests")

    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)
    yield runtime


@pytest.fixture
def temporary_serato_root(tmp_path) -> Path:
    root = tmp_path / "temporary Serato root"
    (root / "SubCrates").mkdir(parents=True)
    (root / "database V2").write_bytes(b"database sentinel")
    (root / "preferences.json").write_text('{"temporary": true}', encoding="utf-8")
    return root


@pytest.fixture
def temporary_subcrates(temporary_serato_root: Path) -> Path:
    return temporary_serato_root / "SubCrates"


@pytest.fixture
def temporary_audio_file(tmp_path) -> Path:
    path = tmp_path / "music library" / "Beyonc\u00e9 Test Song.mp3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"disposable audio bytes; not a real recording")
    return path


@pytest.fixture
def temporary_music_library(tmp_path) -> Path:
    library = tmp_path / "music library"
    (library / "nested").mkdir(parents=True)
    (library / "Artist - One.mp3").write_bytes(b"one")
    (library / "nested" / "Artist - Two.flac").write_bytes(b"two")
    (library / "ignore.txt").write_text("not audio", encoding="utf-8")
    return library


@pytest.fixture
def valid_crate_file(temporary_subcrates: Path, temporary_audio_file: Path) -> Path:
    path = temporary_subcrates / "Valid.crate"
    path.write_bytes(
        b"vrsn\x00\x00\x00\x00"
        + serialize_serato_track_record(str(temporary_audio_file.resolve()).lstrip("/"))
    )
    return path


@pytest.fixture
def malformed_crate_file(temporary_subcrates: Path) -> Path:
    path = temporary_subcrates / "Malformed.crate"
    path.write_bytes(b"vrsn\x00\x00\x00\x00otrk\x00\x00\x00\x09ptrk\x00\x00\x00\x01/")
    return path


@pytest.fixture
def temporary_model_file(tmp_path) -> Path:
    model_file = tmp_path / "fixture-model.pkl"
    joblib.dump({"model": FixturePredictionModel(), "feature_columns": ["feature"]}, model_file)
    return model_file


@pytest.fixture
def temporary_feature_cache(tmp_path) -> Path:
    cache = tmp_path / "feature-cache.json"
    cache.write_text('{"features": []}', encoding="utf-8")
    return cache


@pytest.fixture
def temporary_backup_folder(tmp_path) -> Path:
    backup = tmp_path / "backups"
    backup.mkdir()
    return backup


@pytest.fixture
def temporary_application_configuration(isolate_filesystem_network_and_runtime) -> dict[str, str]:
    return {
        "pending_queue": config.PENDING_QUEUE_PATH,
        "processed_index": config.PROCESSED_INDEX_PATH,
    }


@pytest.fixture
def isolated_streamlit_session_state() -> dict[str, object]:
    return {
        "pred_df": None,
        "queue_df": None,
        "manual_crate_selections": {},
        "queue_crate_selections": {},
    }


@pytest.fixture
def mock_prediction_model() -> FixturePredictionModel:
    return FixturePredictionModel()


@pytest.fixture
def mock_probability_outputs() -> list[float]:
    return [0.5, 0.3, 0.2]


@pytest.fixture
def mock_watcher_queue(temporary_application_configuration) -> list[dict]:
    return [
        {
            "path": "/temporary/queued.mp3",
            "Song Title": "Queued",
            "Suggested Crate": "House%%Club",
            "Confidence": 0.5,
            "_top1_crate": "House%%Club",
            "_top1_prob": 0.5,
        }
    ]


@pytest.fixture
def mock_tag_writer() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_serato_writer() -> MagicMock:
    return MagicMock()
