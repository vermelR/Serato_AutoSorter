"""Side-effect orchestration for approved tracks, independent of Streamlit."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from serato_ai.core.models import ApplyRequest, ApprovedTrack
from serato_ai.services.crate_assignment_service import CrateAssignmentService


def apply_approved_rows(
    approved: pd.DataFrame,
    serato_root: str | Path,
    dry_run: bool,
    make_backup: bool,
    *,
    tag_writer: Callable[..., object],
    crate_writer: Callable[..., Iterable[object]],
) -> tuple[list[dict], list[dict]]:
    """Tag each physical track once and write every selected raw crate pair.

    Writer results remain separate from tag results, so a tag failure cannot
    conceal an unsafe/failed crate write (or vice versa).
    """
    tracks = tuple(
        ApprovedTrack(
            path=str(row["path"]),
            final_crates=tuple(row.get("Final Crates", [])),
            genre=str(row.get("Genre", "")),
            year=str(row.get("Year", "")),
        )
        for _, row in approved.iterrows()
    )
    response = CrateAssignmentService(
        tag_writer=tag_writer,
        crate_writer=crate_writer,
    ).apply(ApplyRequest(tracks, Path(serato_root), dry_run, make_backup))
    return [asdict(result) for result in response.tag_results], [asdict(result) for result in response.crate_results]
