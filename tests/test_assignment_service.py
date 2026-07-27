from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serato_ai.core.models import ApprovedTrack, ApplyRequest, CrateWriteResult, TagWriteResult
from serato_ai.services.crate_assignment_service import CrateAssignmentService


pytestmark = pytest.mark.unit


def test_assignment_service_tags_once_expands_final_crates_and_returns_typed_response() -> None:
    tag_writer = MagicMock(return_value=TagWriteResult("/tmp/one.mp3", True))
    crate_writer = MagicMock(return_value=[
        CrateWriteResult("House%%Club", "/tmp/one.mp3", True, True, "added"),
        CrateWriteResult("House%%Deep", "/tmp/one.mp3", True, True, "added"),
    ])
    request = ApplyRequest(
        (
            ApprovedTrack("/tmp/one.mp3", ("House%%Club", "House%%Deep", "House%%Club"), "House", "2024"),
            ApprovedTrack("/tmp/one.mp3", ("House%%Deep",), "Ignored", "1999"),
        ),
        Path("/temporary/Serato"),
        dry_run=False,
        make_backup=True,
        allowed_crates=("House%%Club", "House%%Deep"),
    )
    response = CrateAssignmentService(tag_writer=tag_writer, crate_writer=crate_writer).apply(request)

    tag_writer.assert_called_once_with("/tmp/one.mp3", genre="House", year="2024")
    assert crate_writer.call_args.kwargs["assignments"] == [
        ("House%%Club", "/tmp/one.mp3"), ("House%%Deep", "/tmp/one.mp3"),
    ]
    assert response.summary is not None and response.summary.all_succeeded
    assert response.crate_results[0].crate_name == "House%%Club"


def test_assignment_service_is_shared_for_manual_and_watcher_requests() -> None:
    writer = MagicMock(return_value=[CrateWriteResult("House%%Club", "/tmp/song.mp3", True, False, "dry_run")])
    service = CrateAssignmentService(tag_writer=MagicMock(), crate_writer=writer)
    manual = ApplyRequest((ApprovedTrack("/tmp/song.mp3", ("House%%Club",)),), Path("/tmp/root"), True, True)
    watcher = ApplyRequest((ApprovedTrack("/tmp/song.mp3", ("House%%Club",)),), Path("/tmp/root"), True, True)
    manual_response = service.apply(manual)
    watcher_response = service.apply(watcher)
    assert manual_response == watcher_response
    assert writer.call_count == 2


def test_assignment_service_rejects_missing_or_disallowed_final_crates_before_side_effects() -> None:
    tag_writer = MagicMock()
    crate_writer = MagicMock()
    response = CrateAssignmentService(tag_writer=tag_writer, crate_writer=crate_writer).apply(
        ApplyRequest(
            (ApprovedTrack("/tmp/song.mp3", ("Hip Hop%%Open",)),),
            Path("/tmp/root"), False, True, allowed_crates=("House%%Club",),
        )
    )
    assert not response.is_valid
    assert response.validation_issues[0].code == "missing_final_crates"
    tag_writer.assert_not_called()
    crate_writer.assert_not_called()
