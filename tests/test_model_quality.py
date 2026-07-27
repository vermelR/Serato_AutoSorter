"""Deterministic M5 quality, leakage, threshold, and history coverage."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier

from serato_ai.core.models import (
    DatasetFingerprint,
    EvaluationArtifact,
    EvaluationConfiguration,
    ModelQualitySnapshot,
    QualityGateResult,
    ThresholdConfiguration,
    TrackTrainingRecord,
)
from serato_ai.core.quality_rules import (
    SplitRecord,
    deterministic_multilabel_split,
    preprocessing_scope_is_safe,
    prediction_quality_state,
    summarize_split,
)
from serato_ai.infrastructure.application_data import ApplicationDataPaths
from serato_ai.infrastructure.model_quality_store import ModelQualityStore
from serato_ai.infrastructure.personal_model_store import PersonalModelStore
from serato_ai.services.model_activation_service import ModelActivationService
from serato_ai.services.model_evaluation_service import ModelEvaluationService
from serato_ai.services.model_health_service import ModelHealthService
from serato_ai.services.model_quality_service import ModelQualityService
from serato_ai.services.model_service import ModelService
from serato_ai.services.training_service import TrainingService
from serato_ai.core.models import TrainingDatasetSummary


pytestmark = [
    pytest.mark.unit,
    pytest.mark.model_quality,
    pytest.mark.metrics,
    pytest.mark.leakage,
    pytest.mark.calibration,
    pytest.mark.thresholds,
    pytest.mark.candidate_comparison,
]


CLASSES = ("House%%Club", "House%%Deep", "Hip Hop%%Open", "Latin%%Afro", "Techno%%Peak")


def _record(index: int, labels: tuple[str, ...], *, fingerprint: str = "") -> TrackTrainingRecord:
    return TrackTrainingRecord(
        canonical_id=f"canonical-{index}", track_path=f"/temporary/{index}.mp3", fingerprint=fingerprint or f"fingerprint-{index}",
        labels=labels, original_labels=labels, features=(float(index % 3), float(index), 1.0, 0.5),
        feature_version="audio-v1-16", validation_status="ready",
    )


def _records(count: int = 18) -> tuple[TrackTrainingRecord, ...]:
    labels: list[tuple[str, ...]] = []
    for index in range(count):
        values = [CLASSES[index % 3]]
        if index % 4 == 0:
            values.append(CLASSES[3])
        if index % 5 == 0:
            values.append(CLASSES[4])
        labels.append(tuple(values))
    return tuple(_record(index, value) for index, value in enumerate(labels))


def _quality_service() -> ModelQualityService:
    return ModelQualityService(EvaluationConfiguration(
        random_seed=7, threshold_minimum_support=2, per_crate_minimum_support=1,
        low_confidence_probability=0.45, low_confidence_margin=0.08,
    ))


def _evaluation_inputs() -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0],
    ])
    probabilities = np.asarray([
        [0.90, 0.10, 0.05, 0.05, 0.02],
        [0.20, 0.85, 0.10, 0.04, 0.03],
        [0.10, 0.20, 0.88, 0.04, 0.08],
        [0.82, 0.10, 0.12, 0.78, 0.05],
        [0.08, 0.10, 0.79, 0.08, 0.72],
        [0.12, 0.80, 0.08, 0.05, 0.04],
    ])
    return probabilities, truth


def test_grouped_multilabel_split_is_deterministic_and_prevents_identity_or_fingerprint_leakage() -> None:
    records = list(_records())
    records.append(_record(99, (CLASSES[0], CLASSES[1]), fingerprint=records[0].fingerprint))
    first = deterministic_multilabel_split(records, random_seed=7)
    second = deterministic_multilabel_split(records, random_seed=7)

    assert first == second
    assert first.summary.leakage_free
    identities = first.summary.train_track_ids + first.summary.validation_track_ids + first.summary.test_track_ids
    assert len(identities) == len(set(identities))
    grouped = next(item for item in first.train + first.validation + first.test if 0 in item.record_indexes)
    assert len(records) - 1 in grouped.record_indexes
    assert {CLASSES[0], CLASSES[1]}.issubset(grouped.labels)


def test_small_dataset_reports_limited_validation_only_evidence() -> None:
    split = deterministic_multilabel_split(_records(8), random_seed=7)
    assert not split.summary.final_test_available
    assert split.summary.test_count == 0
    assert any("too small" in warning for warning in split.summary.warnings)


def test_explicit_leakage_checks_reject_cross_split_and_duplicate_evaluation_rows() -> None:
    duplicate = SplitRecord("same-track", "same-file", (CLASSES[0],), (0,))
    summary = summarize_split((duplicate,), (duplicate,), (), 7)
    assert not summary.leakage_free
    assert any("crosses" in warning for warning in summary.warnings)
    assert preprocessing_scope_is_safe("train_only")
    assert preprocessing_scope_is_safe("not_applicable_no_fitted_preprocessing")
    assert not preprocessing_scope_is_safe("train_and_test")


def test_metrics_cover_multilabel_ranking_calibration_and_hierarchy() -> None:
    service = _quality_service()
    probabilities, truth = _evaluation_inputs()
    thresholds = service.tune_thresholds(probabilities, truth, CLASSES)
    result = service.evaluate(
        probabilities, truth, CLASSES, thresholds, track_ids=tuple(f"track-{index}" for index in range(len(truth))),
        training_support={crate: 5 for crate in CLASSES}, validation_support={crate: 3 for crate in CLASSES}, test_support={crate: 3 for crate in CLASSES},
    )

    assert result.multi_label_metrics.micro_f1 == pytest.approx(1.0)
    assert result.multi_label_metrics.hamming_loss == pytest.approx(0.0)
    assert result.ranking_metrics.top1_hit_rate == pytest.approx(1.0)
    assert result.ranking_metrics.recall_at_3 == pytest.approx(1.0)
    assert result.ranking_metrics.label_ranking_average_precision == pytest.approx(1.0)
    assert result.calibration_metrics.brier_score is not None
    assert result.calibration_metrics.expected_calibration_error is not None
    assert result.calibration_metrics.reliability_bins
    assert result.hierarchy_metrics.top_level_accuracy == pytest.approx(1.0)
    assert result.hierarchy_metrics.cross_category_error_count == 0
    assert all(metric.threshold == thresholds.threshold_for(metric.raw_name) for metric in result.per_crate_metrics)


def test_invalid_probabilities_and_duplicate_evaluation_ids_are_rejected() -> None:
    service = _quality_service()
    probabilities, truth = _evaluation_inputs()
    with pytest.raises(ValueError, match="between zero and one"):
        service.evaluate(probabilities * 2, truth, CLASSES, ThresholdConfiguration())
    with pytest.raises(ValueError, match="unique"):
        service.evaluate(probabilities, truth, CLASSES, ThresholdConfiguration(), track_ids=("same",) * len(truth))


def test_threshold_tuning_is_bounded_per_crate_and_uses_validation_inputs_only() -> None:
    service = _quality_service()
    probabilities, truth = _evaluation_inputs()
    thresholds = service.tune_thresholds(probabilities, truth, CLASSES)
    assert thresholds.source_split == "validation"
    assert thresholds.minimum_threshold <= thresholds.global_threshold <= thresholds.maximum_threshold
    assert thresholds.per_crate
    assert all(thresholds.minimum_threshold <= value <= thresholds.maximum_threshold for _, value in thresholds.per_crate)


def test_low_confidence_states_are_structured_and_do_not_remove_manual_choices() -> None:
    config = _quality_service().configuration
    thresholds = ThresholdConfiguration(global_threshold=0.60)
    assert prediction_quality_state((), thresholds, config).prediction_quality == "no_eligible_suggestion"
    below = prediction_quality_state(((CLASSES[0], 0.50),), thresholds, config)
    assert below.needs_review and "threshold" in below.review_reason
    close = prediction_quality_state(((CLASSES[0], 0.90), (CLASSES[1], 0.85)), thresholds, config)
    assert close.needs_review and "close" in close.review_reason
    supported = prediction_quality_state(((CLASSES[0], 0.90), (CLASSES[1], 0.20)), thresholds, config, support_by_crate={CLASSES[0]: 8})
    assert supported.prediction_quality == "high_confidence"


def test_candidate_comparison_and_gate_allow_improvement_but_reject_regression() -> None:
    service = _quality_service()
    probabilities, truth = _evaluation_inputs()
    thresholds = service.tune_thresholds(probabilities, truth, CLASSES)
    good = service.evaluate(probabilities, truth, CLASSES, thresholds, track_ids=tuple(f"track-{index}" for index in range(len(truth))))
    poor_probabilities = np.full_like(probabilities, 0.10)
    poor = service.evaluate(poor_probabilities, truth, CLASSES, ThresholdConfiguration(global_threshold=0.50), track_ids=tuple(f"track-{index}" for index in range(len(truth))))
    improved = service.compare(poor, good, champion_name="baseline", challenger_name="candidate", comparison_split="test", coverage_preserved=True)
    regressed = service.compare(good, poor, champion_name="active", challenger_name="candidate", comparison_split="test", coverage_preserved=True)

    assert service.quality_gate(good, improved, split_is_valid=True, first_time_model=True).passed
    assert not service.quality_gate(poor, regressed, split_is_valid=True, first_time_model=False).passed


def test_quality_history_round_trip_redacts_audio_paths_and_preserves_rejection(tmp_path) -> None:
    paths = ApplicationDataPaths(tmp_path / "quality-data")
    store = ModelQualityStore(paths)
    artifact = EvaluationArtifact(
        artifact_id="candidate-1", created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=DatasetFingerprint("fingerprint", 3, 2, (CLASSES[0], CLASSES[1])),
        split_summary=deterministic_multilabel_split(_records(12), random_seed=7).summary,
        configuration=_quality_service().configuration, model_algorithm="test-model",
        quality_gate=QualityGateResult(False, ("Macro F1 regressed.",)),
    )
    store.save(ModelQualitySnapshot("candidate-1", artifact, "rejected", ("Macro F1 regressed.",)))

    loaded = store.latest_rejected()
    assert loaded is not None
    assert loaded.artifact.dataset_fingerprint.value == "fingerprint"
    assert loaded.rejection_reasons == ("Macro F1 regressed.",)
    assert "/temporary/" not in str(store.export())


def test_training_persists_quality_snapshot_thresholds_and_safe_activation(tmp_path) -> None:
    records = tuple(
        TrackTrainingRecord(
            canonical_id=f"track-{index}", track_path=f"/temporary/{index}.mp3", fingerprint=f"fingerprint-{index}",
            labels=(CLASSES[index % 2],), original_labels=(CLASSES[index % 2],),
            features=(float(index % 2), float(index % 2), 1.0, 0.0), feature_version="audio-v1-16", validation_status="ready",
        ) for index in range(18)
    )
    paths = ApplicationDataPaths(tmp_path / "application-data")
    quality_store = ModelQualityStore(paths)
    models = PersonalModelStore(paths)
    service = TrainingService(
        quality_store=quality_store, random_seed=7, candidate_limit=1,
        evaluator=ModelEvaluationService(minimum_top3=0.0, minimum_micro_f1=0.0),
    )
    result = service.train(TrainingDatasetSummary(records=records), configuration_fingerprint="config", library_fingerprint="library")

    assert result.succeeded
    assert result.bundle["quality_configuration"]["threshold_configuration"]["source_split"] == "validation"
    assert result.metadata is not None and result.metadata.quality_snapshot_id
    snapshot = quality_store.latest_for(result.metadata.quality_snapshot_id)
    assert snapshot is not None and snapshot.status == "candidate"
    assert snapshot.artifact.split_summary.final_test_available
    assert snapshot.artifact.quality_gate.passed
    assert snapshot.artifact.model_card is not None
    assert snapshot.artifact.preprocessing_scope == "not_applicable_no_fitted_preprocessing"
    assert {item.name for item in snapshot.artifact.feature_ablations} >= {"all_audio_features", "rhythm_energy_subset", "without_accepted_feedback"}
    assert "/temporary/" not in str(quality_store.export())

    activation = ModelActivationService(store=models, model_service=ModelService(personal_store=models), quality_store=quality_store).activate(result)
    assert activation.activated
    assert quality_store.latest_for(result.metadata.quality_snapshot_id).status == "active"
    health = ModelHealthService(model_store=models, quality_store=quality_store).summary()
    assert health.quality_snapshot is not None


def test_rejected_training_candidate_stays_in_quality_history_without_activation(tmp_path) -> None:
    records = tuple(
        TrackTrainingRecord(
            canonical_id=f"track-{index}", track_path=f"/temporary/{index}.mp3", fingerprint=f"fingerprint-{index}",
            labels=(CLASSES[index % 2],), original_labels=(CLASSES[index % 2],),
            features=(float(index % 2), float(index % 2), 1.0, 0.0), feature_version="audio-v1-16", validation_status="ready",
        ) for index in range(18)
    )
    paths = ApplicationDataPaths(tmp_path / "application-data")
    quality_store = ModelQualityStore(paths)
    models = PersonalModelStore(paths)
    service = TrainingService(
        quality_store=quality_store, random_seed=7,
        estimator_factory=lambda: OneVsRestClassifier(DummyClassifier(strategy="constant", constant=0)),
        evaluator=ModelEvaluationService(minimum_top3=0.0, minimum_micro_f1=0.0),
    )
    result = service.train(TrainingDatasetSummary(records=records), configuration_fingerprint="config", library_fingerprint="library")

    assert not result.succeeded
    assert quality_store.latest_rejected() is not None
    assert models.active_path() is None
