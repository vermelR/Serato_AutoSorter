"""M6 compact storage, bounded work, and output-equivalence coverage."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from benchmarks.m6_benchmarks import PROFILES, run_profile
from phase4_engine import propose_crates_for_files
from serato_ai.core.dataframes import merge_prediction_page, paginate_prediction_frame
from serato_ai.core.models import (
    DatasetFingerprint,
    DatasetSplitSummary,
    EvaluationConfiguration,
    EvaluationArtifact,
    MetadataEnrichment,
    MetadataProviderResult,
    ModelEvaluation,
    ModelQualitySnapshot,
    ModelVersionMetadata,
    PerCrateQualityMetric,
    PredictionEvaluationRecord,
    TrackTrainingRecord,
    TrainingDatasetSummary,
    TrainingResult,
)
from serato_ai.core.storage_rules import (
    BUNDLE_SCHEMA_VERSION,
    StorageBudgets,
    assess_model_size,
)
from serato_ai.infrastructure.application_data import ApplicationDataPaths
from serato_ai.infrastructure.feature_cache_store import CachedFeatures, FeatureCacheStore
from serato_ai.infrastructure.personal_model_store import PersonalModelStore
from serato_ai.infrastructure.model_quality_store import ModelQualityStore
from serato_ai.infrastructure.queue_store import QueueStore
from serato_ai.infrastructure.scan_index_store import ScanIndexStore
from serato_ai.infrastructure.training_cache_store import TrainingCacheStore
from serato_ai.services.background_job_service import BackgroundJobManager
from serato_ai.services.feature_extraction_service import FeatureExtractionService
from serato_ai.services.library_scan_service import LibraryScanService
from serato_ai.services.legacy_compact_migration_service import LegacyCompactMigrationService
from serato_ai.services.model_activation_service import ModelActivationService
from serato_ai.services.model_evaluation_service import ModelEvaluationService
from serato_ai.services.model_service import ModelService, clear_model_cache
from serato_ai.services.storage_management_service import StorageManagementService
from serato_ai.services.training_service import MultiLabelCrateModel, TrainingService
from serato_crate import serialize_serato_track_record


pytestmark = [
    pytest.mark.unit,
    pytest.mark.performance,
    pytest.mark.model_storage,
    pytest.mark.model_size,
    pytest.mark.incremental_scan,
    pytest.mark.background_jobs,
    pytest.mark.batch_prediction,
    pytest.mark.migration,
]


CLASSES = ("House%%Club", "House%%Deep")


class CompactFixtureModel:
    classes_ = np.asarray(CLASSES)

    def __init__(self):
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += 1
        values = np.asarray(frame, dtype=float)
        first = np.clip(values[:, 0] / 200.0, 0.05, 0.95)
        return np.column_stack([first, 1.0 - first])


class CountingCandidateEstimator:
    fit_calls = 0

    def fit(self, features, truth):
        type(self).fit_calls += 1
        self.output_count_ = np.asarray(truth).shape[1]
        return self

    def predict_proba(self, features):
        values = np.asarray(features, dtype=float)
        first = np.where(values[:, 0] < 0.5, 0.95, 0.05)
        return np.column_stack([first, 1.0 - first])


class CancellingCandidateEstimator(CountingCandidateEstimator):
    def __init__(self, cancel_event: Event):
        self.cancel_event = cancel_event

    def fit(self, features, truth):
        result = super().fit(features, truth)
        self.cancel_event.set()
        return result


class MetadataStub:
    def read_embedded(self, _path):
        return MetadataProviderResult(status="missing")

    def enrich(self, path, **_kwargs):
        return MetadataEnrichment(title=path.stem, artist="Fixture")


def _metadata(version: str = "m6-test") -> ModelVersionMetadata:
    return ModelVersionMetadata(
        version=version,
        trained_at=datetime.now(UTC).isoformat(),
        feature_schema_version="audio-v1-16",
        model_format_version=4,
        class_names=CLASSES,
        eligible_crates=CLASSES,
        dataset_summary=TrainingDatasetSummary(records=(
            TrackTrainingRecord(
                "private-id",
                "/temporary/private-track.mp3",
                "fingerprint",
                (CLASSES[0],),
                features=(1.0,),
                validation_status="ready",
            ),
        )),
        evaluation=ModelEvaluation(quality="good", passed_quality_gate=True),
        configuration_fingerprint="configuration",
        library_fingerprint="library",
        estimator_type="fixture",
    )


def test_legacy_multiclass_forest_dense_class_values_dominate_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    rows, classes = 1200, 60
    features = rng.normal(size=(rows, 16)).astype(np.float32)
    labels = np.arange(rows) % classes
    forest = RandomForestClassifier(n_estimators=5, random_state=7, n_jobs=1)
    forest.fit(features, labels)
    forest_path = tmp_path / "legacy.joblib"
    joblib.dump({"model": forest, "feature_columns": list(range(16))}, forest_path)
    dense_values = sum(tree.tree_.value.nbytes for tree in forest.estimators_)
    all_tree_arrays = dense_values + sum(
        tree.tree_.children_left.nbytes
        + tree.tree_.children_right.nbytes
        + tree.tree_.feature.nbytes
        + tree.tree_.threshold.nbytes
        + tree.tree_.impurity.nbytes
        + tree.tree_.n_node_samples.nbytes
        + tree.tree_.weighted_n_node_samples.nbytes
        for tree in forest.estimators_
    )

    binary_truth = np.column_stack([labels == 0, labels == 1]).astype(int)
    compact = MultiLabelCrateModel(
        OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=100)).fit(
            features,
            binary_truth,
        ),
        CLASSES,
    )
    compact_path = tmp_path / "compact.joblib"
    joblib.dump({"model": compact, "feature_columns": list(range(16))}, compact_path, compress=3)

    assert dense_values / all_tree_arrays > 0.85
    assert compact_path.stat().st_size < forest_path.stat().st_size / 10


def test_storage_budgets_warn_review_block_and_allow_explicit_override() -> None:
    budgets = StorageBudgets(100, 200, 300, 400)
    assert assess_model_size(50, budgets).status == "compact"
    assert assess_model_size(150, budgets).status == "preferred_target_exceeded"
    assert assess_model_size(250, budgets).status == "warning"
    assert not assess_model_size(350, budgets).automatic_activation_allowed
    assert not assess_model_size(450, budgets).automatic_activation_allowed
    override = StorageBudgets(100, 200, 300, 400, allow_developer_override=True)
    assert assess_model_size(450, override).automatic_activation_allowed


def test_inference_bundle_is_versioned_compressed_and_excludes_training_state(tmp_path: Path) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    store = PersonalModelStore(paths)
    bundle = {
        "model": CompactFixtureModel(),
        "feature_columns": ["feature_0"],
        "prediction_semantics": "independent_multilabel",
        "dataset_summary": {"private": True},
        "feature_cache": {"large": True},
        "quality_history": [{"large": True}],
    }
    path = store.save_candidate("m6-test", bundle, _metadata())
    persisted = joblib.load(path)
    stored_metadata = store.metadata_for_version("m6-test")

    assert persisted["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert persisted["compression_method"] == "zlib-3"
    assert persisted["artifact_size_bytes"] == path.stat().st_size
    assert persisted["inference_metadata"]["artifact_size_bytes"] == path.stat().st_size
    assert "dataset_summary" not in persisted
    assert "feature_cache" not in persisted
    assert "quality_history" not in persisted
    assert "model_metadata" not in persisted
    assert stored_metadata is not None
    assert stored_metadata.artifact_size_bytes == path.stat().st_size
    assert stored_metadata.dataset_summary.records[0].track_path == "/temporary/private-track.mp3"


def test_compressed_bundle_size_metadata_reaches_an_exact_fixed_point(
    tmp_path: Path,
) -> None:
    for index in range(16):
        version = f"fixed-point-{'x' * index}"
        store = PersonalModelStore(
            ApplicationDataPaths(tmp_path / f"application-data-{'y' * index}")
        )
        path = store.save_candidate(
            version,
            {
                "model": CompactFixtureModel(),
                "feature_columns": ["feature_0"],
                "prediction_semantics": "independent_multilabel",
            },
            _metadata(version),
        )
        persisted = joblib.load(path)
        metadata = store.metadata_for_version(version)

        assert persisted["artifact_size_bytes"] == path.stat().st_size
        assert (
            persisted["inference_metadata"]["artifact_size_bytes"]
            == path.stat().st_size
        )
        assert metadata is not None
        assert metadata.artifact_size_bytes == path.stat().st_size


def test_evaluation_records_are_external_compressed_and_history_stays_compact(
    tmp_path: Path,
) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    store = ModelQualityStore(paths)
    records = tuple(
        PredictionEvaluationRecord(
            track_id=f"hashed-{index}",
            true_labels=(CLASSES[index % 2],),
            ranked_labels=CLASSES if index % 2 == 0 else tuple(reversed(CLASSES)),
            probabilities=(0.9, 0.1) if index % 2 == 0 else (0.2, 0.8),
            predicted_labels=(CLASSES[index % 2],),
        )
        for index in range(100)
    )
    artifact = EvaluationArtifact(
        artifact_id="external-evaluation",
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=DatasetFingerprint("fingerprint", 100, 2, CLASSES),
        split_summary=DatasetSplitSummary("fixture", 42, leakage_free=True),
        configuration=EvaluationConfiguration(),
        model_algorithm="fixture",
        per_crate_metrics=tuple(
            PerCrateQualityMetric(raw_name=name) for name in CLASSES
        ),
        prediction_records=records,
    )
    store.save(ModelQualitySnapshot("external-evaluation", artifact, "candidate"))

    snapshot = store.latest_for("external-evaluation")
    assert snapshot is not None
    assert snapshot.artifact.prediction_records == ()
    assert snapshot.artifact.prediction_record_count == 100
    assert snapshot.artifact.prediction_records_reference
    assert store.evaluation_records_path("external-evaluation").is_file()
    assert paths.quality_history_file.stat().st_size < 20_000
    restored = store.load_prediction_records("external-evaluation")
    assert len(restored) == len(records)
    assert restored[0].true_labels == records[0].true_labels
    assert restored[0].probabilities == pytest.approx(records[0].probabilities)
    assert (
        store.evaluation_records_path("external-evaluation").stat().st_mode
        & 0o777
    ) == 0o600


def test_identical_candidate_evaluation_is_reused_only_for_matching_fingerprints(
    tmp_path: Path,
) -> None:
    records = tuple(
        TrackTrainingRecord(
            canonical_id=f"track-{index}",
            track_path=f"/disposable/{index}.mp3",
            fingerprint=f"fingerprint-{index}",
            labels=(CLASSES[index % 2],),
            original_labels=(CLASSES[index % 2],),
            features=(float(index % 2), 1.0, 0.0, 0.0),
            feature_version="audio-v1-16",
            validation_status="ready",
        )
        for index in range(24)
    )
    paths = ApplicationDataPaths(tmp_path / "app")
    cache = TrainingCacheStore(paths)
    service = TrainingService(
        quality_store=ModelQualityStore(paths),
        training_cache=cache,
        estimator_factory=CountingCandidateEstimator,
        candidate_limit=1,
        random_seed=7,
        evaluator=ModelEvaluationService(
            minimum_top3=0.0,
            minimum_micro_f1=0.0,
        ),
    )
    dataset = TrainingDatasetSummary(records=records)
    CountingCandidateEstimator.fit_calls = 0

    first = service.train(
        dataset,
        configuration_fingerprint="configuration-a",
        library_fingerprint="library",
    )
    first_calls = CountingCandidateEstimator.fit_calls
    second = service.train(
        dataset,
        configuration_fingerprint="configuration-a",
        library_fingerprint="library",
    )
    second_calls = CountingCandidateEstimator.fit_calls - first_calls
    third = service.train(
        dataset,
        configuration_fingerprint="configuration-b",
        library_fingerprint="library",
    )
    third_calls = (
        CountingCandidateEstimator.fit_calls - first_calls - second_calls
    )

    assert first.succeeded and second.succeeded and third.succeeded
    # Every run still performs the documented feature ablation. The matching
    # second run skips the bounded candidate fit/evaluation itself.
    assert first_calls == 2
    assert second_calls == 1
    assert third_calls == 2
    assert tuple(paths.training_cache.rglob("candidates/*.joblib"))


def test_training_cancellation_never_produces_an_activatable_candidate(
    tmp_path: Path,
) -> None:
    records = tuple(
        TrackTrainingRecord(
            canonical_id=f"track-{index}",
            track_path=f"/disposable/{index}.mp3",
            fingerprint=f"fingerprint-{index}",
            labels=(CLASSES[index % 2],),
            features=(float(index % 2), 1.0, 0.0, 0.0),
            feature_version="audio-v1-16",
            validation_status="ready",
        )
        for index in range(24)
    )
    paths = ApplicationDataPaths(tmp_path / "app")
    cancelled = Event()
    service = TrainingService(
        quality_store=ModelQualityStore(paths),
        estimator_factory=lambda: CancellingCandidateEstimator(cancelled),
        candidate_limit=1,
        random_seed=7,
        evaluator=ModelEvaluationService(
            minimum_top3=0.0,
            minimum_micro_f1=0.0,
        ),
    )

    result = service.train(
        TrainingDatasetSummary(records=records),
        configuration_fingerprint="configuration",
        library_fingerprint="library",
        cancel_event=cancelled,
    )
    models = PersonalModelStore(paths)
    activation = ModelActivationService(
        store=models,
        model_service=ModelService(personal_store=models),
        quality_store=ModelQualityStore(paths),
    ).activate(result)

    assert not result.succeeded
    assert "cancelled" in " ".join(result.diagnostics).casefold()
    assert not activation.activated
    assert models.active_path() is None


def test_model_service_loads_once_per_immutable_file_version(tmp_path: Path) -> None:
    clear_model_cache()
    path = tmp_path / "model.joblib"
    joblib.dump({"model": CompactFixtureModel(), "feature_columns": ["feature_0"]}, path)
    calls = {"count": 0}

    def loader(value):
        calls["count"] += 1
        return joblib.load(value)

    service_one = ModelService(loader=loader)
    service_one.load(str(path))
    service_one.load(str(path))
    assert calls["count"] == 1

    time.sleep(0.002)
    joblib.dump({"model": CompactFixtureModel(), "feature_columns": ["feature_0"]}, path)
    service_one.load(str(path))
    assert calls["count"] == 2


def test_failed_active_load_falls_back_to_previous_without_changing_pointers(
    tmp_path: Path,
) -> None:
    clear_model_cache()
    paths = ApplicationDataPaths(tmp_path / "app")
    store = PersonalModelStore(paths)
    for version in ("previous", "active"):
        candidate = store.save_candidate(
            version,
            {
                "model": CompactFixtureModel(),
                "feature_columns": ["feature_0"],
                "prediction_semantics": "independent_multilabel",
                "sentinel": version,
            },
            _metadata(version),
        )
        store.activate(candidate, _metadata(version))
    active = store.active_path()
    previous = store.previous_path()
    assert active is not None and previous is not None
    active.write_bytes(b"corrupt candidate")

    loaded = ModelService(personal_store=store).load()

    assert loaded["sentinel"] == "previous"
    assert store.active_path() == active
    assert store.previous_path() == previous


def test_oversized_candidate_is_blocked_and_active_previous_are_protected(
    tmp_path: Path,
) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    budgets = StorageBudgets(1_000, 2_000, 5_000, 10_000)
    store = PersonalModelStore(paths, storage_budgets=budgets)
    activation = ModelActivationService(
        store=store,
        model_service=ModelService(personal_store=store),
        quality_store=ModelQualityStore(paths),
    )
    compact = TrainingResult(
        True,
        bundle={
            "model": CompactFixtureModel(),
            "feature_columns": ["feature_0"],
            "prediction_semantics": "independent_multilabel",
        },
        metadata=_metadata("compact-active"),
    )
    assert activation.activate(compact).activated
    active_before = store.active_path()
    oversized = TrainingResult(
        True,
        bundle={
            "model": CompactFixtureModel(),
            "feature_columns": ["feature_0"],
            "prediction_semantics": "independent_multilabel",
            "padding": np.random.default_rng(7).bytes(16_384),
        },
        metadata=_metadata("oversized-rejected"),
    )
    blocked = activation.activate(oversized)
    assert not blocked.activated
    assert store.active_path() == active_before
    assert store.candidate_path("oversized-rejected").is_file()


def test_retention_cleanup_never_removes_active_previous_or_legacy(
    tmp_path: Path,
) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    store = PersonalModelStore(paths)
    bundle = {
        "model": CompactFixtureModel(),
        "feature_columns": ["feature_0"],
        "prediction_semantics": "independent_multilabel",
    }
    activated = []
    for version in ("one", "two", "three"):
        candidate = store.save_candidate(version, bundle, _metadata(version))
        activated.append(store.activate(candidate, _metadata(version))[1])
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"recoverable")
    store.record_legacy_reference(legacy)
    storage = StorageManagementService(paths, model_store=store)

    preview = storage.cleanup_obsolete_model_versions(confirmed=False)
    assert not preview.removed_files
    result = storage.cleanup_obsolete_model_versions(confirmed=True)

    assert str(activated[0]) in result.removed_files
    assert not activated[0].exists()
    assert store.active_path() == activated[2] and store.active_path().is_file()
    assert store.previous_path() == activated[1] and store.previous_path().is_file()
    assert legacy.read_bytes() == b"recoverable"


def test_explicit_cleanup_removes_only_rejected_temporary_and_obsolete_reports(
    tmp_path: Path,
) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    store = PersonalModelStore(paths)
    quality = ModelQualityStore(paths)
    rejected_version = "rejected-cleanup"
    rejected = store.save_candidate(
        rejected_version,
        {
            "model": CompactFixtureModel(),
            "feature_columns": ["feature_0"],
            "prediction_semantics": "independent_multilabel",
        },
        _metadata(rejected_version),
    )
    artifact = EvaluationArtifact(
        artifact_id=rejected_version,
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=DatasetFingerprint("cleanup", 1, 2, CLASSES),
        split_summary=DatasetSplitSummary("fixture", 42),
        configuration=EvaluationConfiguration(),
        model_algorithm="fixture",
    )
    quality.save(
        ModelQualitySnapshot(
            rejected_version,
            artifact,
            "rejected",
            ("fixture rejection",),
        )
    )
    paths.ensure()
    temporary = paths.root / "interrupted.tmp"
    temporary.write_bytes(b"temporary")
    obsolete_evaluation = paths.evaluations / "orphan.npz"
    obsolete_evaluation.write_bytes(b"orphan")
    obsolete_backup = paths.reports / "report.bak"
    obsolete_backup.write_bytes(b"backup")
    storage = StorageManagementService(
        paths,
        model_store=store,
        quality_store=quality,
    )

    assert storage.summary().reclaimable_bytes >= (
        rejected.stat().st_size
        + temporary.stat().st_size
        + obsolete_evaluation.stat().st_size
        + obsolete_backup.stat().st_size
    )
    assert not storage.cleanup_rejected_candidates().removed_files
    assert rejected.is_file()
    assert not storage.cleanup_temporary_files().removed_files
    assert temporary.is_file()
    assert not storage.cleanup_obsolete_reports().removed_files
    assert obsolete_evaluation.is_file() and obsolete_backup.is_file()

    assert storage.cleanup_rejected_candidates(
        confirmed=True
    ).removed_files == (str(rejected),)
    assert str(temporary) in storage.cleanup_temporary_files(
        confirmed=True
    ).removed_files
    removed_reports = set(
        storage.cleanup_obsolete_reports(confirmed=True).removed_files
    )
    assert removed_reports == {
        str(obsolete_evaluation),
        str(obsolete_backup),
    }


def test_batch_prediction_matches_individual_and_reduces_estimator_calls(monkeypatch, tmp_path: Path) -> None:
    files = [tmp_path / f"track-{index}.mp3" for index in range(4)]
    for path in files:
        path.write_bytes(b"fixture")
    features = {
        str(path): [80.0 + index * 20]
        for index, path in enumerate(files)
    }
    monkeypatch.setattr("phase4_engine.extract_audio_features", lambda path: features[path])
    model = CompactFixtureModel()
    bundle = {
        "model": model,
        "feature_columns": ["feature_0"],
        "prediction_semantics": "independent_multilabel",
    }
    batch, failures = propose_crates_for_files(
        bundle,
        files,
        topk=2,
        metadata_service=MetadataStub(),
    )
    batch_calls = model.calls
    individual_frames = []
    model.calls = 0
    for path in files:
        frame, individual_failures = propose_crates_for_files(
            bundle,
            [path],
            topk=2,
            metadata_service=MetadataStub(),
        )
        assert individual_failures.empty
        individual_frames.append(frame)
    individual = pd.concat(individual_frames, ignore_index=True)

    assert failures.empty
    assert batch_calls == 1
    assert model.calls == len(files)
    assert list(batch["Suggested Crate"]) == list(individual["Suggested Crate"])
    assert np.allclose(batch["Confidence"], individual["Confidence"])
    assert list(batch["Review Reason"]) == list(individual["Review Reason"])
    assert list(batch["Prediction Quality"]) == list(individual["Prediction Quality"])
    assert np.allclose(batch["Threshold Used"], individual["Threshold Used"])
    assert list(batch["_allowed_crates"]) == list(individual["_allowed_crates"])
    for rank in (1, 2):
        assert list(batch[f"_top{rank}_crate"]) == list(
            individual[f"_top{rank}_crate"]
        )
        assert np.allclose(
            batch[f"_top{rank}_prob"],
            individual[f"_top{rank}_prob"],
        )


def test_sqlite_feature_cache_incremental_lookup_compaction_and_corruption(tmp_path: Path) -> None:
    paths = ApplicationDataPaths(tmp_path / "app")
    cache = FeatureCacheStore(paths)
    cache.put(CachedFeatures("one", "v1", (1.25, 2.5)))
    cache.put(CachedFeatures("two", "v1", (3.5, 4.5)))
    assert cache.get("one", "v1").values == pytest.approx((1.25, 2.5))
    assert cache.get("missing", "v1") is None
    assert cache.stats().hit_rate == pytest.approx(0.5)
    assert cache.delete_not_in({"one"}, "v1") == 1
    cache.compact()
    assert cache.count() == 1

    cache.cache_file.write_bytes(b"corrupt sqlite")
    assert cache.get("one", "v1") is None
    cache.put(CachedFeatures("recovered", "v1", (9.0,)))
    assert cache.get("recovered", "v1") is not None
    assert tuple(cache.cache_file.parent.glob("*.corrupt-*"))


def test_incremental_crate_and_music_scan_reuses_unchanged_and_matches_full(tmp_path: Path) -> None:
    root = tmp_path / "Serato"
    subcrates = root / "SubCrates"
    music = tmp_path / "Music"
    subcrates.mkdir(parents=True)
    music.mkdir()
    track = music / "one.mp3"
    track.write_bytes(b"one")
    crate = subcrates / "House.crate"
    crate.write_bytes(
        b"vrsn\x00\x00\x00\x00"
        + serialize_serato_track_record(str(track.resolve()).lstrip("/"))
    )
    paths = ApplicationDataPaths(tmp_path / "app")
    service = LibraryScanService(scan_index=ScanIndexStore(paths))

    first = service.scan(root, (str(music),))
    second = service.scan(root, (str(music),))
    assert first.scan_summary.crate_files_parsed == 1
    assert second.scan_summary.crate_files_parsed == 0
    assert second.scan_summary.crate_files_reused == 1
    assert second.library_fingerprint == first.library_fingerprint

    crate.write_bytes(b"malformed")
    malformed = service.scan(root, (str(music),))
    assert malformed.crate_records[0].track_paths == first.crate_records[0].track_paths
    assert any("prior valid state" in warning for warning in malformed.warnings)

    crate.write_bytes(
        b"vrsn\x00\x00\x00\x00"
        + serialize_serato_track_record(str(track.resolve()).lstrip("/"))
    )
    incremental = service.scan(root, (str(music),))
    full = service.scan(root, (str(music),), full_rescan=True)
    assert incremental.library_fingerprint == full.library_fingerprint
    assert incremental.crate_records[0].track_paths == full.crate_records[0].track_paths
    assert [(item.status, item.resolved_path) for item in incremental.track_matches] == [
        (item.status, item.resolved_path) for item in full.track_matches
    ]


def test_incremental_music_scan_handles_add_modify_move_delete_and_drive_reconnect(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Serato"
    subcrates = root / "SubCrates"
    music = tmp_path / "Music"
    nested = music / "Nested"
    subcrates.mkdir(parents=True)
    nested.mkdir(parents=True)
    first_track = nested / "one.mp3"
    second_track = nested / "two.mp3"
    first_track.write_bytes(b"one")
    second_track.write_bytes(b"two")
    crate = subcrates / "House.crate"
    crate.write_bytes(
        b"vrsn\x00\x00\x00\x00"
        + serialize_serato_track_record(str(first_track.resolve()).lstrip("/"))
    )
    paths = ApplicationDataPaths(tmp_path / "app")
    service = LibraryScanService(scan_index=ScanIndexStore(paths))

    initial = service.scan(root, (str(music),), full_rescan=True)
    unchanged = service.scan(root, (str(music),))
    assert unchanged.scan_summary.directories_scanned == 0
    assert unchanged.scan_summary.music_files_reused == 2

    first_track.write_bytes(b"one-modified")
    modified = service.scan(root, (str(music),))
    assert modified.scan_summary.music_files_modified == 1

    added_track = nested / "three.mp3"
    added_track.write_bytes(b"three")
    added = service.scan(root, (str(music),))
    assert added.scan_summary.music_files_added == 1
    assert added.scan_summary.directories_scanned >= 1

    moved_track = nested / "two-renamed.mp3"
    second_track.rename(moved_track)
    moved = service.scan(root, (str(music),))
    assert moved.scan_summary.music_files_moved == 1
    assert moved.scan_summary.music_files_removed == 0

    added_track.unlink()
    deleted = service.scan(root, (str(music),))
    assert deleted.scan_summary.music_files_removed == 1

    disconnected = tmp_path / "Music-disconnected"
    music.rename(disconnected)
    sleeping = service.scan(root, (str(music),))
    assert sleeping.scan_summary.unavailable_folders == 1
    assert sleeping.scan_summary.music_files_seen == 2
    assert sleeping.library_fingerprint == deleted.library_fingerprint

    disconnected.rename(music)
    reconnected = service.scan(root, (str(music),))
    clean = LibraryScanService(
        scan_index=ScanIndexStore(
            paths,
            paths.root / "clean-reconnect-index.json",
        )
    ).scan(root, (str(music),), full_rescan=True)
    assert reconnected.scan_summary.unavailable_folders == 0
    assert reconnected.library_fingerprint == clean.library_fingerprint
    assert initial.library_fingerprint != reconnected.library_fingerprint


def test_scan_cancellation_preserves_previous_index(tmp_path: Path) -> None:
    root = tmp_path / "Serato"
    subcrates = root / "SubCrates"
    music = tmp_path / "Music"
    subcrates.mkdir(parents=True)
    music.mkdir()
    track = music / "one.mp3"
    track.write_bytes(b"one")
    crate = subcrates / "House.crate"
    crate.write_bytes(
        b"vrsn\x00\x00\x00\x00"
        + serialize_serato_track_record(str(track.resolve()).lstrip("/"))
    )
    paths = ApplicationDataPaths(tmp_path / "app")
    index = ScanIndexStore(paths)
    service = LibraryScanService(scan_index=index)
    first = service.scan(root, (str(music),))
    prior = paths.scan_index_file.read_bytes()
    cancelled = Event()
    cancelled.set()

    with pytest.raises(InterruptedError, match="prior scan index"):
        service.scan(
            root,
            (str(music),),
            full_rescan=True,
            cancel_event=cancelled,
        )

    assert paths.scan_index_file.read_bytes() == prior
    assert service.scan(root, (str(music),)).library_fingerprint == first.library_fingerprint


def test_queue_store_collapses_duplicate_bursts_and_is_bounded(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "pending.jsonl", tmp_path / "processed.json", maximum_pending=2)
    assert store.append_pending({"path": "/tmp/one.mp3"})
    assert not store.append_pending({"path": "/tmp/one.mp3"})
    assert store.append_pending({"path": "/tmp/two.mp3"})
    assert not store.append_pending({"path": "/tmp/three.mp3"})
    assert store.count_pending() == 2
    assert [row["path"] for row in store.load_pending(limit=1, offset=1)] == [
        "/tmp/two.mp3"
    ]


def test_queue_store_pages_searches_filters_and_persists_review_edits(
    tmp_path: Path,
) -> None:
    store = QueueStore(
        tmp_path / "pending.sqlite3",
        tmp_path / "processed.json",
    )
    house_signature = QueueStore.filter_signature(CLASSES)
    rows = [
        {
            "path": "/tmp/alpha.mp3",
            "Song Title": "Alpha",
            "Needs Review": True,
            "_allowed_crates": list(CLASSES),
        },
        {
            "path": "/tmp/beta.mp3",
            "Song Title": "Beta",
            "Needs Review": False,
            "_allowed_crates": list(CLASSES),
        },
        {
            "path": "/tmp/other.mp3",
            "Song Title": "Other",
            "Needs Review": True,
            "_allowed_crates": ["Other%%Crate"],
        },
    ]
    assert store.append_many(rows) == (True, True, True)
    filters = (house_signature,)

    assert store.count_pending(filter_signatures=filters) == 2
    assert [
        row["path"]
        for row in store.load_pending(
            limit=1,
            offset=1,
            filter_signatures=filters,
        )
    ] == ["/tmp/beta.mp3"]
    assert store.count_pending(
        search="beta",
        filter_signatures=filters,
    ) == 1
    assert store.count_pending(
        view="Low confidence only",
        filter_signatures=filters,
    ) == 1

    edited = dict(rows[1])
    edited["Approve"] = True
    edited["Final Crates"] = list(CLASSES)
    assert store.update_pending([edited]) == 1
    assert store.count_pending(
        view="Approved only",
        filter_signatures=filters,
    ) == 1
    assert store.count_pending(
        view="Pending only",
        filter_signatures=filters,
    ) == 1
    restored = store.load_pending(
        view="Approved only",
        filter_signatures=filters,
    )
    assert restored[0]["Final Crates"] == list(CLASSES)


def test_queue_store_recovers_corrupt_local_database_without_losing_new_row(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "pending.sqlite3"
    queue_path.write_bytes(b"\xffnot-a-database")
    store = QueueStore(queue_path, tmp_path / "processed.json")

    assert store.append_pending({"path": "/tmp/recovered.mp3"})
    assert store.load_pending() == [{"path": "/tmp/recovered.mp3"}]
    assert tuple(tmp_path.glob("pending.sqlite3.legacy-jsonl*"))


def test_background_jobs_report_checkpoint_and_cancel_safely(tmp_path: Path) -> None:
    manager = BackgroundJobManager(
        ApplicationDataPaths(tmp_path / "app"),
        jobs_file=tmp_path / "jobs.json",
    )
    started = Event()

    def target(context):
        context.checkpoint("features:10", resume_supported=True)
        started.set()
        for index in range(100):
            context.progress("Extracting", index, 100)
            time.sleep(0.001)

    job_id = manager.submit("feature_extraction", target, resume_supported=True)
    assert started.wait(1)
    assert manager.cancel(job_id)
    snapshot = manager.wait(job_id, timeout=2)
    manager.shutdown()
    assert snapshot.status == "cancelled"
    assert snapshot.checkpoint == "features:10"
    assert snapshot.resume_supported
    assert snapshot.elapsed_seconds >= 0


def test_background_jobs_report_failure_eta_resume_and_bound_history(
    tmp_path: Path,
) -> None:
    manager = BackgroundJobManager(
        ApplicationDataPaths(tmp_path / "app"),
        jobs_file=tmp_path / "jobs.json",
        history_limit=10,
    )

    def failing(context):
        context.progress("Evaluating", 2, 4)
        context.checkpoint("candidate:2", resume_supported=True)
        raise RuntimeError("synthetic failure")

    failed_id = manager.submit("evaluation", failing, resume_supported=True)
    failed = manager.wait(failed_id, timeout=2)
    assert failed.status == "failed"
    assert failed.estimated_seconds_remaining is not None
    assert "synthetic failure" in failed.diagnostics[-1]

    seen_checkpoints = []

    def resumed(context):
        seen_checkpoints.append(context.resume_checkpoint)
        context.progress("Evaluating", 4, 4)
        return "complete"

    resumed_id = manager.resume(failed_id, resumed)
    assert resumed_id is not None
    resumed_snapshot = manager.wait(resumed_id, timeout=2)
    assert resumed_snapshot.status == "completed"
    assert seen_checkpoints == ["candidate:2"]
    assert manager.result(resumed_id) == "complete"

    submitted = []
    for index in range(12):
        job_id = manager.submit(f"small-{index}", lambda _context: index)
        submitted.append(job_id)
        assert manager.wait(job_id, timeout=2).status == "completed"
    manager.shutdown()
    retained = manager.snapshots()
    assert len(retained) == 10
    assert submitted[-1] in {item.job_id for item in retained}


def test_parallel_extraction_matches_sequential_and_uses_bounded_workers(tmp_path: Path) -> None:
    files = [tmp_path / f"{index}.mp3" for index in range(6)]
    for index, path in enumerate(files):
        path.write_bytes(str(index).encode())
    records = tuple(
        TrackTrainingRecord(
            f"id-{index}",
            str(path),
            "",
            (CLASSES[index % 2],),
        )
        for index, path in enumerate(files)
    )
    extractor = lambda path: [float(Path(path).stem), 1.0]
    sequential = FeatureExtractionService(
        cache=FeatureCacheStore(ApplicationDataPaths(tmp_path / "seq")),
        extractor=extractor,
        metadata_reader=lambda _path: {},
        resource_mode="low_resource",
    )
    parallel = FeatureExtractionService(
        cache=FeatureCacheStore(ApplicationDataPaths(tmp_path / "parallel")),
        extractor=extractor,
        metadata_reader=lambda _path: {},
        resource_mode="fast",
        maximum_workers=3,
    )
    seq_rows = sequential.extract(records)[0]
    parallel_rows = parallel.extract(records)[0]
    assert [row.features for row in seq_rows] == [row.features for row in parallel_rows]
    assert parallel.maximum_workers == 3


def test_parallel_extraction_isolates_failure_and_cancellation_keeps_cache(
    tmp_path: Path,
) -> None:
    files = [tmp_path / f"parallel-{index}.mp3" for index in range(8)]
    for index, path in enumerate(files):
        path.write_bytes(str(index).encode())
    records = tuple(
        TrackTrainingRecord(
            f"id-{index}",
            str(path),
            "",
            (CLASSES[index % 2],),
        )
        for index, path in enumerate(files)
    )

    def partly_failing(path):
        if Path(path).stem == "parallel-2":
            raise ValueError("isolated fixture failure")
        return [float(Path(path).stem.rsplit("-", 1)[1]), 1.0]

    isolated = FeatureExtractionService(
        cache=FeatureCacheStore(ApplicationDataPaths(tmp_path / "isolated")),
        extractor=partly_failing,
        metadata_reader=lambda _path: {},
        resource_mode="fast",
        maximum_workers=99,
    )
    isolated_rows, _, isolated_extracted, isolated_failed, warnings = (
        isolated.extract(records)
    )
    assert isolated.maximum_workers == 4
    assert isolated_extracted == 7
    assert isolated_failed == 1
    assert isolated_rows[2].validation_status == "feature_failed"
    assert "isolated fixture failure" in warnings[0]
    assert [Path(row.track_path).name for row in isolated_rows] == [
        path.name for path in files
    ]

    cache = FeatureCacheStore(ApplicationDataPaths(tmp_path / "cancelled"))
    cancel = Event()

    def slow_extract(path):
        time.sleep(0.01)
        return [float(Path(path).stem.rsplit("-", 1)[1]), 1.0]

    def cancel_after_first(progress):
        if progress.completed:
            cancel.set()

    cancelled_service = FeatureExtractionService(
        cache=cache,
        extractor=slow_extract,
        metadata_reader=lambda _path: {},
        resource_mode="fast",
        maximum_workers=3,
    )
    cancelled_rows, _, extracted, failed, _ = cancelled_service.extract(
        records,
        progress=cancel_after_first,
        cancel_event=cancel,
    )
    assert failed == 0
    assert 1 <= extracted <= 3
    assert cache.count() == extracted
    assert sum(
        row.validation_status == "cancelled" for row in cancelled_rows
    ) == len(records) - extracted

    completed = tuple(
        row for row in cancelled_rows if row.validation_status == "ready"
    )
    reuse_only = FeatureExtractionService(
        cache=cache,
        extractor=lambda _path: (_ for _ in ()).throw(
            AssertionError("completed cache entry was not reused")
        ),
        metadata_reader=lambda _path: {},
        resource_mode="low_resource",
    )
    _, reused, re_extracted, re_failed, _ = reuse_only.extract(completed)
    assert (reused, re_extracted, re_failed) == (extracted, 0, 0)


def test_large_table_pagination_filter_and_merge_preserve_final_crates() -> None:
    frame = pd.DataFrame([
        {
            "path": f"/tmp/{index}.mp3",
            "Song Title": f"Song {index}",
            "Artist": "Artist",
            "Approve": index == 0,
            "Needs Review": index % 2 == 0,
            "Final Crates": [CLASSES[0]],
        }
        for index in range(120)
    ])
    page = paginate_prediction_frame(frame, page=2, page_size=25)
    assert len(page.frame) == 25
    assert page.total_pages == 5
    low = paginate_prediction_frame(frame, view="Low confidence only", page_size=50)
    assert low.filtered_rows == 60
    edited = page.frame.copy()
    edited.at[edited.index[0], "Final Crates"] = [CLASSES[0], CLASSES[1]]
    merged = merge_prediction_page(frame, edited)
    assert merged.loc[edited.index[0], "Final Crates"] == [CLASSES[0], CLASSES[1]]


def test_legacy_csv_migration_builds_quality_gated_compact_model_and_preserves_source(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"legacy artifact sentinel")
    csv_path = tmp_path / "features.csv"
    pd.DataFrame([
        {
            "title": f"Track {index}",
            "path": f"/offline/source/{index}.mp3",
            "crate": CLASSES[index % 2],
            "bpm": float(index % 2),
            "brightness": float(index % 2),
            "energy": 1.0,
            **{f"mfcc_{feature}": float(index % 2) for feature in range(1, 14)},
        }
        for index in range(24)
    ]).to_csv(csv_path, index=False)
    paths = ApplicationDataPaths(tmp_path / "app")
    models = PersonalModelStore(paths)
    migration = LegacyCompactMigrationService(model_store=models)

    report = migration.migrate(legacy, csv_path)

    assert report.legacy_preserved
    assert legacy.read_bytes() == b"legacy artifact sentinel"
    assert report.candidate_result.succeeded
    assert report.activation_result.activated
    assert models.active_path() is not None
    assert models.active_path().stat().st_size < 500 * 1024**2
    assert models.legacy_reference()["path"] == str(legacy.resolve())
    assert ModelService(personal_store=models).resolve_model_path(str(legacy)) == str(models.active_path())


def test_small_repeatable_benchmark_covers_required_bounded_work(tmp_path: Path) -> None:
    result = run_profile(PROFILES["small"], tmp_path / "benchmark")

    assert result["profile"] == {"name": "small", "tracks": 500, "crates": 20}
    assert result["model"]["disk_load_calls"] == 1
    assert result["prediction"]["estimator_calls_for_batch"] == 1
    assert result["feature_cache"]["hits"] == 500
    assert result["watcher_burst"]["stored"] == 500
    assert result["scanning"]["full_crates_parsed"] == 20
    assert result["scanning"]["unchanged_crates_parsed"] == 0
    assert result["scanning"]["unchanged_crates_reused"] == 20
    assert result["scanning"]["unchanged_directories_enumerated"] == 0
    assert result["scanning"]["incremental_equals_clean_full"]
    assert result["table"]["records"] == 500
    assert result["table"]["rendered_rows"] == 50
    assert result["memory"]["estimated_live_bytes"] > 0
