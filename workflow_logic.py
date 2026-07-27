"""Compatibility facade for workflow rules now owned by ``serato_ai.core``."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pandas as pd

from serato_ai.core.assignment_utils import expand_assignments, unique_tracks
from serato_ai.core.crate_filters import filter_crate_selections, format_crate_label
from serato_ai.core.dataframes import (
    crate_result_column_options,
    format_crate_suggestions,
    normalize_crate_selections,
    selection_path_key,
)
from serato_ai.core.models import ApprovedTrack, CrateWriteResult
from serato_ai.core.result_summary import summarize_crate_results


def crate_widget_key(scope: str, path: object) -> str:
    digest = hashlib.sha256(selection_path_key(path).encode("utf-8")).hexdigest()
    return f"{scope}_crates_{digest}"


def build_apply_plan(
    approved: pd.DataFrame,
    path_key: Callable[[object], str],
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    tracks = [
        ApprovedTrack(
            path=str(row["path"]),
            final_crates=tuple(normalize_crate_selections(row.get("Final Crates", []))),
            genre=str(row.get("Genre", "")),
            year=str(row.get("Year", "")),
        )
        for _, row in approved.iterrows()
    ]
    tag_jobs = [
        {"path": track.path, "genre": track.genre, "year": track.year}
        for track in unique_tracks(tracks, path_key=path_key)
    ]
    assignments = [
        (assignment.crate_name, assignment.track_path)
        for assignment in expand_assignments(tracks, path_key=path_key)
    ]
    return tag_jobs, assignments


def rows_missing_final_crates(approved: pd.DataFrame) -> list[int]:
    return [
        int(index)
        for index, row in approved.iterrows()
        if not normalize_crate_selections(row.get("Final Crates", []))
    ]


def crate_write_outcome(succeeded: int, total: int) -> tuple[str, str, bool]:
    placeholder = [CrateWriteResult("", "", index < succeeded) for index in range(total)]
    summary = summarize_crate_results(placeholder)
    return summary.level, summary.message, summary.all_succeeded


def should_mark_queue_reviewed(*, dry_run: bool, crate_writes_succeeded: bool) -> bool:
    return not dry_run and crate_writes_succeeded


def restrict_final_crates(approved: pd.DataFrame, allowed_crates: list[str]) -> pd.DataFrame:
    result = approved.copy()
    result["Final Crates"] = result["Final Crates"].apply(
        lambda values: filter_crate_selections(values, allowed_crates)
    )
    return result


__all__ = [
    "build_apply_plan", "crate_result_column_options", "crate_widget_key",
    "crate_write_outcome", "format_crate_label", "format_crate_suggestions",
    "normalize_crate_selections", "restrict_final_crates", "rows_missing_final_crates",
    "selection_path_key", "should_mark_queue_reviewed",
]
