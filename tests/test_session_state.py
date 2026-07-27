from __future__ import annotations

import pandas as pd
import pytest

from serato_ai.ui.session_state import (
    MANUAL_CRATE_SELECTIONS,
    MANUAL_PREDICTIONS,
    QUEUE_CRATE_SELECTIONS,
    WATCHER_QUEUE,
    get_manual_crate_selections,
    get_queue_crate_selections,
    initialize_session_state,
    reset_manual_prediction_state,
    reset_queue_state,
    set_manual_predictions,
)


pytestmark = pytest.mark.unit


def test_session_state_initialization_and_manual_queue_isolation() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    get_manual_crate_selections(state)["/tmp/song.mp3"] = ["House%%Club"]
    get_queue_crate_selections(state)["/tmp/song.mp3"] = ["Hip Hop%%Open"]
    assert state[MANUAL_CRATE_SELECTIONS] != state[QUEUE_CRATE_SELECTIONS]


def test_resetting_manual_predictions_does_not_erase_queue_state() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    set_manual_predictions(state, pd.DataFrame([{"path": "/tmp/manual.mp3"}]), pd.DataFrame(), ("House%%Club",))
    state[WATCHER_QUEUE] = pd.DataFrame([{"path": "/tmp/queue.mp3"}])
    reset_manual_prediction_state(state)
    assert state[MANUAL_PREDICTIONS] is None
    assert state[WATCHER_QUEUE].iloc[0]["path"] == "/tmp/queue.mp3"


def test_resetting_queue_does_not_erase_manual_crate_selections() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    state[MANUAL_CRATE_SELECTIONS] = {"/tmp/manual.mp3": ["House%%Club"]}
    reset_queue_state(state)
    assert state[MANUAL_CRATE_SELECTIONS] == {"/tmp/manual.mp3": ["House%%Club"]}
    assert isinstance(state[WATCHER_QUEUE], pd.DataFrame) and state[WATCHER_QUEUE].empty
