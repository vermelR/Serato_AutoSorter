from __future__ import annotations

import pandas as pd
import pytest

from workflow_logic import (
    build_apply_plan,
    crate_widget_key,
    crate_write_outcome,
    crate_result_column_options,
    format_crate_label,
    format_crate_suggestions,
    normalize_crate_selections,
    rows_missing_final_crates,
    restrict_final_crates,
    selection_path_key,
    should_mark_queue_reviewed,
)


pytestmark = pytest.mark.unit


def test_final_crate_defaults_and_deduplication_preserve_raw_names() -> None:
    top_suggestion = "House%%Club"
    selections = normalize_crate_selections([top_suggestion, top_suggestion, "House%%Deep"])
    assert selections == ["House%%Club", "House%%Deep"]
    assert format_crate_label(selections[0]) == "House › Club"
    assert "›" not in selections[0]


def test_final_crate_selection_state_is_scoped_and_widget_keys_are_stable() -> None:
    path = "file:///temporary/music/My%20Song.mp3"
    normalized = selection_path_key(path)
    assert normalized == "/temporary/music/My Song.mp3"
    assert crate_widget_key("manual", path) == crate_widget_key("manual", normalized)
    assert crate_widget_key("manual", path) != crate_widget_key("queue", path)

    manual = {normalized: ["House%%Club"]}
    watcher = {normalized: ["Hip Hop%%Open Format"]}
    manual[normalized].append("House%%Deep")
    assert manual[normalized] == ["House%%Club", "House%%Deep"]
    assert watcher[normalized] == ["Hip Hop%%Open Format"]


def test_apply_plan_uses_only_final_crates_tags_once_and_keeps_raw_names() -> None:
    approved = pd.DataFrame(
        [
            {
                "path": "/tmp/one.mp3",
                "Genre": "House",
                "Year": "2024",
                "Suggested Crate": "Ignore%%This",
                "_top1_crate": "Ignore%%This",
                "Final Crates": ["House%%Club", "House%%Deep", "House%%Club"],
            },
            {
                "path": "/tmp/one.mp3",
                "Genre": "Other row must not retag",
                "Year": "1999",
                "Final Crates": ["House%%Deep"],
            },
        ]
    )
    tag_jobs, assignments = build_apply_plan(approved, selection_path_key)

    assert tag_jobs == [{"path": "/tmp/one.mp3", "genre": "House", "year": "2024"}]
    assert assignments == [
        ("House%%Club", "/tmp/one.mp3"),
        ("House%%Deep", "/tmp/one.mp3"),
    ]
    assert all("›" not in crate for crate, _ in assignments)


def test_only_approved_rows_require_final_crates() -> None:
    approved = pd.DataFrame(
        [
            {"path": "/tmp/approved.mp3", "Final Crates": []},
            {"path": "/tmp/approved-ok.mp3", "Final Crates": ["House%%Club"]},
        ],
        index=[10, 11],
    )
    assert rows_missing_final_crates(approved) == [10]
    # A denied row never reaches the approved frame, so it needs no crate.
    assert rows_missing_final_crates(pd.DataFrame(columns=["path", "Final Crates"])) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.03, "3.0%"), (0.09, "9.0%"), (0.825, "82.5%")],
)
def test_suggestions_format_raw_probability_once(value: float, expected: str) -> None:
    row = pd.Series({"_top1_crate": "House%%Club", "_top1_prob": value})
    display = format_crate_suggestions(row, 5, ["House%%Club"])
    assert display == f"1. House › Club ({expected})"
    assert row["_top1_prob"] == value


def test_suggestion_lines_keep_rank_skip_missing_and_cap_to_available_classes() -> None:
    row = pd.Series(
        {
            "_top1_crate": "House%%Club",
            "_top1_prob": 0.825,
            "_top2_crate": "Hip Hop%%Open Format",
            "_top2_prob": 0.09,
            "_top3_crate": None,
            "_top3_prob": None,
        }
    )
    display = format_crate_suggestions(row, 10, ["House%%Club", "Hip Hop%%Open Format"])
    assert display.splitlines() == [
        "1. House › Club (82.5%)",
        "2. Hip Hop › Open Format (9.0%)",
    ]
    assert "%%" not in display


@pytest.mark.parametrize(
    ("succeeded", "total", "level", "all_succeeded"),
    [
        (1, 1, "success", True),
        (1, 2, "warning", False),
        (0, 2, "error", False),
        (0, 0, "error", False),
    ],
)
def test_result_reporting_never_claims_all_failed_assignments_succeeded(
    succeeded: int, total: int, level: str, all_succeeded: bool,
) -> None:
    actual_level, message, actual_all_succeeded = crate_write_outcome(succeeded, total)
    assert actual_level == level
    assert actual_all_succeeded is all_succeeded
    if succeeded == total and total:
        assert message.startswith("All")
    else:
        assert not message.startswith("All")


@pytest.mark.parametrize(
    ("dry_run", "writes_succeeded", "should_remove"),
    [(True, True, False), (True, False, False), (False, False, False), (False, True, True)],
)
def test_queue_rows_are_only_marked_reviewed_after_live_success(
    dry_run: bool, writes_succeeded: bool, should_remove: bool,
) -> None:
    assert should_mark_queue_reviewed(
        dry_run=dry_run, crate_writes_succeeded=writes_succeeded
    ) is should_remove


def test_crate_result_table_contract_keeps_success_read_only_and_errors_visible() -> None:
    options = crate_result_column_options()
    assert options["success"] == {"label": "Success", "disabled": True}
    assert options["status"] == {"label": "Status"}
    assert options["error"] == {"label": "Error", "width": "large"}


def test_apply_boundary_blocks_disallowed_final_crates() -> None:
    approved = pd.DataFrame([
        {"path": "/tmp/one.mp3", "Final Crates": ["House%%Club", "Hip Hop%%Open Format"]},
    ])
    filtered = restrict_final_crates(approved, ["House%%Club"])
    assert filtered.iloc[0]["Final Crates"] == ["House%%Club"]
