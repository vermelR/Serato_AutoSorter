"""Temporary-only tests for onboarding, training, activation, feedback, and health."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from serato_ai.core.models import (
    ApplyResponse, ApprovedTrack, CrateTrainingSummary, CrateWriteResult, ModelEvaluation, OperationSummary,
    ModelActivationResult, OnboardingStatus, SeratoCrateRecord, SeratoLibraryScanResult, TagWriteResult, TrackMatch, TrackTrainingRecord,
    TrainingDatasetSummary, TrainingProgress, TrainingResult,
)
from serato_ai.core.confidence import filter_and_rank_independent_probabilities
from serato_ai.infrastructure.application_data import ApplicationDataPaths
from serato_ai.infrastructure.feature_cache_store import FeatureCacheStore
from serato_ai.infrastructure.feedback_store import FeedbackStore
from serato_ai.infrastructure.onboarding_store import OnboardingStore
from serato_ai.infrastructure.personal_model_store import PersonalModelStore
from serato_ai.services.feature_extraction_service import FeatureExtractionService
from serato_ai.services.feedback_service import FeedbackService
from serato_ai.services.library_scan_service import LibraryScanService
from serato_ai.services.model_activation_service import ModelActivationService
from serato_ai.services.model_evaluation_service import ModelEvaluationService
from serato_ai.services.model_health_service import ModelHealthService
from serato_ai.services.model_service import ModelService
from serato_ai.services.onboarding_service import OnboardingService
from serato_ai.services.training_dataset_service import TrainingDatasetService
from serato_ai.services.training_service import TrainingService
from serato_ai.core.metadata_rules import METADATA_STRATEGY_TEXT, METADATA_STRATEGY_VERSION
from serato_crate import serialize_serato_track_record


pytestmark = [pytest.mark.unit, pytest.mark.onboarding, pytest.mark.training, pytest.mark.cache, pytest.mark.model_store, pytest.mark.model_health]


def _paths(tmp_path: Path) -> ApplicationDataPaths:
    return ApplicationDataPaths(tmp_path / "SeratoAI application data")


def _scan(audio: Path) -> SeratoLibraryScanResult:
    return SeratoLibraryScanResult(
        serato_root="/temporary/serato",
        crate_records=(
            SeratoCrateRecord("House%%Club", "House › Club", ("House", "Club"), "/temporary/Club.crate", (str(audio),)),
            SeratoCrateRecord("Open%%Dance", "Open › Dance", ("Open", "Dance"), "/temporary/Dance.crate", (str(audio),)),
        ),
        crate_summaries=(
            CrateTrainingSummary("House%%Club", "House › Club", 3, "eligible"),
            CrateTrainingSummary("Open%%Dance", "Open › Dance", 3, "eligible"),
        ),
        track_matches=(TrackMatch(str(audio), str(audio), "matched", (str(audio),), ("House%%Club", "Open%%Dance")),),
        library_fingerprint="fixture-library",
    )


def _dataset() -> TrainingDatasetSummary:
    records = []
    for index in range(10):
        labels = ("House%%Club", "Open%%Dance") if index % 3 == 0 else (("House%%Club",) if index % 2 else ("Open%%Dance",))
        records.append(TrackTrainingRecord(str(index), f"/temporary/{index}.mp3", str(index), labels, features=(float(index % 2), float(index // 2)), feature_version="audio-v1-16", validation_status="ready"))
    return TrainingDatasetSummary(records=tuple(records), matched_tracks=len(records))


def test_library_scan_reads_multiple_raw_crates_without_changing_bytes(temporary_serato_root: Path, temporary_audio_file: Path) -> None:
    first = temporary_serato_root / "SubCrates" / "House%%Club.crate"
    second = temporary_serato_root / "SubCrates" / "Open%%Dance.crate"
    payload = b"vrsn\x00\x00\x00\x00" + serialize_serato_track_record(str(temporary_audio_file.resolve()).lstrip("/"))
    first.write_bytes(payload)
    second.write_bytes(payload)
    before = (first.read_bytes(), second.read_bytes())

    service = LibraryScanService(home=temporary_serato_root.parent)
    result = service.scan(str(temporary_serato_root), (str(temporary_audio_file.parent),), minimum_examples=2)

    assert result.crate_count == 2
    assert {summary.raw_name for summary in result.crate_summaries} == {"House%%Club", "Open%%Dance"}
    assert result.track_matches[0].status == "matched"
    assert result.track_matches[0].crate_names == ("House%%Club", "Open%%Dance")
    assert (first.read_bytes(), second.read_bytes()) == before


def test_detection_validation_and_eligibility_are_safe_and_configurable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    root = home / "Music" / "_Serato_"
    (root / "Subcrates").mkdir(parents=True)
    service = LibraryScanService(home=home, platform="darwin")
    assert service.detected_roots() == (root,)
    assert service.validate_root(root).valid
    assert not service.validate_root(tmp_path / "backup" / "_Serato_").valid

    audio = tmp_path / "Música folder" / "A song.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"x")
    folders, warnings = service.normalize_music_folders((str(audio.parent), str(audio.parent), str(tmp_path / "missing")))
    assert folders == (audio.parent,)
    assert len(warnings) == 1
    assert LibraryScanService(home=home, platform="win32").detected_roots() == (root,)
    assert not service.validate_root(tmp_path / "not-a-folder").valid
    no_subcrates = tmp_path / "custom-serato"
    no_subcrates.mkdir()
    assert "Subcrates" in service.validate_root(no_subcrates).warnings[0]


def test_matcher_reports_ambiguous_missing_and_unsupported_without_filename_guessing(tmp_path: Path) -> None:
    one = tmp_path / "one" / "same.mp3"
    two = tmp_path / "two" / "same.mp3"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_bytes(b"1")
    two.write_bytes(b"2")
    exact, names, _, _ = LibraryScanService._index_audio_files((one.parent, two.parent))
    assert LibraryScanService._resolve_reference("/old/same.mp3", exact, names, (one.parent, two.parent))[0] == "ambiguous"
    assert LibraryScanService._resolve_reference("/old/missing.mp3", exact, names, (one.parent, two.parent))[0] == "missing"
    assert LibraryScanService._resolve_reference("/old/note.txt", exact, names, (one.parent, two.parent))[0] == "unsupported"


def test_feature_cache_reuses_then_invalidates_modified_disposable_audio(tmp_path: Path) -> None:
    audio = tmp_path / "music" / "one.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"one")
    calls: list[str] = []
    cache = FeatureCacheStore(_paths(tmp_path))
    extractor = FeatureExtractionService(cache=cache, extractor=lambda path: calls.append(path) or [1.0, 2.0], metadata_reader=lambda _path: {})
    service = TrainingDatasetService(feature_service=extractor, feedback_store=FeedbackStore(_paths(tmp_path)))

    first = service.build(_scan(audio))
    second = service.build(_scan(audio))
    audio.write_bytes(b"changed")
    third = service.build(_scan(audio))

    assert first.records[0].labels == ("House%%Club", "Open%%Dance")
    assert (first.feature_extracted, second.feature_reused, third.feature_extracted) == (1, 1, 1)
    assert len(calls) == 2
    cache.cache_file.write_text("not JSON", encoding="utf-8")
    fourth = service.build(_scan(audio))
    assert fourth.feature_extracted == 1  # corrupt cache recovers from local source, never tags it
    assert not list(cache.cache_file.parent.glob("*.tmp"))


def test_training_is_multilabel_deterministic_and_activation_is_atomic(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = PersonalModelStore(paths)
    evaluator = ModelEvaluationService(minimum_top3=0.0, minimum_micro_f1=0.0)
    trainer = TrainingService(evaluator=evaluator, minimum_examples=3, random_seed=7)
    first = trainer.train(_dataset(), configuration_fingerprint="config", library_fingerprint="library")
    second = trainer.train(_dataset(), configuration_fingerprint="config", library_fingerprint="library")

    assert first.succeeded and second.succeeded
    assert first.metadata is not None and first.metadata.class_names == ("House%%Club", "Open%%Dance")
    assert first.metadata.metadata_strategy_version == METADATA_STRATEGY_VERSION
    assert first.bundle["prediction_semantics"] == "independent_multilabel"
    activation = ModelActivationService(store=store, model_service=ModelService(personal_store=store)).activate(first)
    assert activation.activated
    assert store.active_path() is not None
    assert ModelService(personal_store=store).crate_options() == ["House%%Club", "Open%%Dance"]

    failed = ModelActivationService(store=store).activate(replace(first, succeeded=False))
    assert not failed.activated
    assert store.active_path() is not None  # failed candidate did not replace the working pointer
    assert not ModelActivationService(store=PersonalModelStore(_paths(tmp_path / "empty"))).restore_previous().activated
    assert first.metadata is not None
    blocked_metadata = replace(first.metadata, evaluation=replace(first.metadata.evaluation, passed_quality_gate=False))
    assert not ModelActivationService(store=store).activate(replace(first, metadata=blocked_metadata)).activated

    second_activation = ModelActivationService(store=store, model_service=ModelService(personal_store=store)).activate(second)
    assert second_activation.activated and ModelActivationService(store=store).restore_previous().activated


def test_independent_multilabel_probabilities_rank_without_renormalizing() -> None:
    suggestions = filter_and_rank_independent_probabilities(
        (("House%%Club", 0.8), ("Open%%Dance", 0.6), ("Inbox", 0.1)), excluded_crates=("Inbox",), topk=2,
    )
    assert [(item.crate_name, item.probability) for item in suggestions] == [("House%%Club", 0.8), ("Open%%Dance", 0.6)]


def test_onboarding_state_legacy_migration_feedback_and_health(tmp_path: Path, temporary_model_file: Path) -> None:
    paths = _paths(tmp_path)
    models = PersonalModelStore(paths)
    state = OnboardingStore(paths)
    onboarding = OnboardingService(store=state, model_store=models, model_service=ModelService(personal_store=models))
    assert onboarding.status().phase == "not_started"
    assert onboarding.status(str(temporary_model_file)).phase == "model_ready"
    started = onboarding.start("/temporary/serato", ("/temporary/music",))
    assert started.phase == "in_progress"
    assert onboarding.pause().phase == "paused"

    response = ApplyResponse(
        tag_results=(TagWriteResult("/temporary/song.mp3", True),),
        crate_results=(CrateWriteResult("House%%Club", "/temporary/song.mp3", True, True, "added"),),
        summary=OperationSummary(1, 1, "success", "ok", True),
    )
    feedback = FeedbackService(store=FeedbackStore(paths), model_store=models)
    tracks = (ApprovedTrack("/temporary/song.mp3", ("House%%Club",)),)
    assert len(feedback.record_successful_apply(tracks, response, source="manual", suggestions={"/temporary/song.mp3": ("House%%Club",)}, dry_run=False)) == 1
    assert len(feedback.record_successful_apply(tracks, response, source="manual", suggestions={}, dry_run=True)) == 0
    health = ModelHealthService(onboarding_store=state, model_store=models, cache=FeatureCacheStore(paths), feedback_store=FeedbackStore(paths)).summary()
    assert health.pending_correction_count == 1
    assert health.metadata_strategy == METADATA_STRATEGY_TEXT


def test_dataset_feedback_precedence_is_not_marked_learned_until_activation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    feedback_store = FeedbackStore(paths)
    feedback = FeedbackService(store=feedback_store, model_store=PersonalModelStore(paths))
    response = ApplyResponse(
        tag_results=(TagWriteResult(str(audio), True),),
        crate_results=(CrateWriteResult("Open%%Dance", str(audio), True, True, "added"),),
        summary=OperationSummary(1, 1, "success", "ok", True),
    )
    feedback.record_successful_apply((ApprovedTrack(str(audio), ("Open%%Dance",)),), response, source="watcher")
    features = FeatureExtractionService(cache=FeatureCacheStore(paths), extractor=lambda _path: [1.0], metadata_reader=lambda _path: {})
    datasets = TrainingDatasetService(feature_service=features, feedback_store=feedback_store)
    dataset = datasets.build(_scan(audio))
    assert dataset.records[0].label_source == "feedback"
    assert dataset.records[0].labels == ("Open%%Dance",)
    assert len(feedback_store.pending()) == 1
    datasets.mark_feedback_included(dataset.feedback_record_ids)
    assert not feedback_store.pending()


class _FakeScanner:
    def __init__(self, scan: SeratoLibraryScanResult):
        self.result = scan

    def detected_roots(self):
        return ()

    def validate_root(self, root):
        from serato_ai.services.library_scan_service import SeratoRootValidation

        return SeratoRootValidation(str(root), True)

    def scan(self, *_args, **_kwargs):
        return self.result


class _FakeDataset:
    def __init__(self):
        self.included = ()

    def build(self, _scan, progress=None):
        if progress:
            progress(TrainingProgress("Building dataset", 1, 1))
        return TrainingDatasetSummary(feedback_record_ids=("feedback-1",))

    def mark_feedback_included(self, values):
        self.included = values


class _FakeTrainer:
    def __init__(self, result):
        self.result = result

    def train(self, *_args, **_kwargs):
        return self.result


class _FakeActivation:
    def __init__(self, result):
        self.result = result

    def activate(self, _candidate):
        return self.result


def test_onboarding_scan_resume_success_and_failed_candidate_states(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    scan = SeratoLibraryScanResult("/temporary/root", library_fingerprint="current")
    dataset = _FakeDataset()
    service = OnboardingService(
        store=OnboardingStore(paths), model_store=PersonalModelStore(paths), scanner=_FakeScanner(scan), dataset_service=dataset,
        training_service=_FakeTrainer(TrainingResult(False, diagnostics=("not enough data",))),
        activation_service=_FakeActivation(ModelActivationResult(True, "v1", message="active")),
    )
    assert service.scan_library("/temporary/root", ("/temporary/music",)).serato_root == "/temporary/root"
    assert service.status().phase == "in_progress"
    result = service.train()
    assert result.activated
    assert service.status().phase == "model_ready"
    assert dataset.included == ("feedback-1",)
    assert service.restart().phase == "not_started"

    failed = OnboardingService(
        store=OnboardingStore(paths), model_store=PersonalModelStore(paths), scanner=_FakeScanner(scan), dataset_service=_FakeDataset(),
        training_service=_FakeTrainer(TrainingResult(False)), activation_service=_FakeActivation(ModelActivationResult(False, message="quality gate")),
    )
    failed.scan_library("/temporary/root", ("/temporary/music",))
    assert not failed.train().activated
    assert failed.status().phase == "failed"


@pytest.mark.integration
def test_disposable_library_onboarding_trains_activates_and_restarts_without_crate_writes(tmp_path: Path) -> None:
    root = tmp_path / "_Serato_"
    subcrates = root / "Subcrates"
    music = tmp_path / "music with spaces"
    subcrates.mkdir(parents=True)
    music.mkdir()
    tracks = []
    for index in range(10):
        path = music / f"track {index}.mp3"
        path.write_bytes(f"audio-{index}".encode())
        tracks.append(path)
    def crate_bytes(indices):
        return b"vrsn\x00\x00\x00\x00" + b"".join(serialize_serato_track_record(str(tracks[index]).lstrip("/")) for index in indices)
    house = subcrates / "House%%Club.crate"
    dance = subcrates / "Open%%Dance.crate"
    house.write_bytes(crate_bytes(range(0, 6)))
    dance.write_bytes(crate_bytes(range(4, 10)))
    before = (house.read_bytes(), dance.read_bytes())

    paths = _paths(tmp_path)
    models = PersonalModelStore(paths)
    features = FeatureExtractionService(
        cache=FeatureCacheStore(paths),
        extractor=lambda path: [float(int(Path(path).stem.split()[-1]) % 2), float(int(Path(path).stem.split()[-1]) // 2)],
        metadata_reader=lambda _path: {},
    )
    datasets = TrainingDatasetService(feature_service=features, feedback_store=FeedbackStore(paths))
    onboarding = OnboardingService(
        store=OnboardingStore(paths), model_store=models, scanner=LibraryScanService(home=tmp_path), dataset_service=datasets,
        training_service=TrainingService(evaluator=ModelEvaluationService(minimum_top3=0.0, minimum_micro_f1=0.0), minimum_examples=3),
        model_service=ModelService(personal_store=models),
    )
    scan = onboarding.scan_library(str(root), (str(music),), minimum_examples=3)
    assert all(match.status == "matched" for match in scan.track_matches)
    activation = onboarding.train()
    assert activation.activated
    assert OnboardingService(store=OnboardingStore(paths), model_store=models, model_service=ModelService(personal_store=models)).status().phase == "model_ready"
    assert ModelService(personal_store=models).crate_options() == ["House%%Club", "Open%%Dance"]
    tracks[0].write_bytes(b"audio changed")
    changed, count, _ = onboarding.update_preview()
    assert changed and count >= 1 and onboarding.status().phase == "update_available"
    assert (house.read_bytes(), dance.read_bytes()) == before
