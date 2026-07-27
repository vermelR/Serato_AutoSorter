from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


pytestmark = pytest.mark.streamlit


def _prediction() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Song Title": "Test",
            "path": "/tmp/test.mp3",
            "Suggested Crate": "House%%Club",
            "Confidence": 0.5,
            "_top1_crate": "House%%Club",
            "_top1_prob": 0.5,
            "_top2_crate": "Hip Hop%%Open Format",
            "_top2_prob": 0.3,
            "_top3_crate": "House%%Deep",
            "_top3_prob": 0.2,
        }
    ])


def test_category_selection_limits_displayed_suggestions_and_final_crates() -> None:
    app = AppTest.from_file("app.py")
    options = ["House%%Club", "Hip Hop%%Open Format", "House%%Deep"]
    app.session_state["crate_options"] = options
    app.session_state["pred_df"] = _prediction()
    app.session_state["manual_prediction_filter_signature"] = tuple(sorted(options))
    app.run(timeout=30)

    category_picker = next(
        widget for widget in app.sidebar.multiselect if widget.label == "Allowed crate categories"
    )
    category_picker.set_value(["House"])
    app.run(timeout=30)

    # A category change clears stale probabilities. Seed an equivalent fresh
    # model row for the selected allow-list and verify the rendered result.
    app.session_state["pred_df"] = _prediction()
    app.session_state["manual_prediction_filter_signature"] = (
        "House%%Club", "House%%Deep",
    )
    app.run(timeout=30)

    frame = app.session_state["pred_df"]
    assert frame.iloc[0]["Top Suggested Crate"] == "House › Club"
    assert "Hip Hop" not in frame.iloc[0]["Crate Suggestions"]
    assert frame.iloc[0]["Final Crates"] == ["House%%Club"]

    final_picker = next(
        widget for widget in app.multiselect if widget.label.startswith("Final Crates:")
    )
    assert final_picker.options == ["House › Club", "House › Deep"]
