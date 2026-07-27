"""Reproducible compact-candidate comparison on saved scalar features."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from collections import Counter
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from serato_ai.core.models import EvaluationConfiguration
from serato_ai.core.quality_rules import deterministic_multilabel_split
from benchmarks.experimental_estimators import (
    SharedNearestPositiveClassifier,
)
from serato_ai.services.compact_estimators import SharedKNNMultiLabelClassifier
from serato_ai.services.legacy_compact_migration_service import LegacyCompactMigrationService
from serato_ai.services.model_quality_service import ModelQualityService
from serato_ai.services.training_service import MultiLabelCrateModel


def compare(
    features_csv: str | Path,
    neighbors: tuple[int, ...],
    distance_powers: tuple[float, ...] = (1.0,),
    probability_scales: tuple[float, ...] = (1.0,),
    nearest_positive_temperatures: tuple[float, ...] = (),
    positive_neighbors: int = 1,
    metrics: tuple[str, ...] = ("euclidean",),
) -> dict:
    dataset = LegacyCompactMigrationService().build_dataset(features_csv)
    counts = Counter(label for record in dataset.records for label in record.labels)
    classes = tuple(sorted(label for label, count in counts.items() if count >= 3))
    allowed = set(classes)
    rows = [
        replace(record, labels=tuple(label for label in record.labels if label in allowed))
        for record in dataset.records
    ]
    rows = [record for record in rows if record.labels]
    split = deterministic_multilabel_split(rows, random_seed=42)
    groups = tuple(split.train + split.validation + split.test)
    collapsed = []
    group_index = {}
    for group in groups:
        source = rows[group.record_indexes[0]]
        collapsed.append(replace(source, labels=group.labels))
        group_index[group.track_id] = len(collapsed) - 1
    indexes = lambda values: [group_index[value.track_id] for value in values]
    train_indexes = indexes(split.train)
    validation_indexes = indexes(split.validation)
    test_indexes = indexes(split.test) or validation_indexes
    matrix = np.asarray([record.features for record in collapsed], dtype=np.float32)
    truth = MultiLabelBinarizer(classes=list(classes)).fit_transform(
        [record.labels for record in collapsed]
    ).astype(np.uint8)
    quality = ModelQualityService(EvaluationConfiguration(random_seed=42))
    results = []
    tracemalloc.start()
    for metric in metrics:
        for count in neighbors:
            for distance_power in distance_powers:
                fit_started = time.perf_counter()
                estimator = SharedKNNMultiLabelClassifier(
                    n_neighbors=count,
                    metric=metric,
                    distance_power=distance_power,
                    probability_scale=1.0,
                )
                estimator.fit(matrix[train_indexes], truth[train_indexes])
                validation_base = estimator.predict_proba(matrix[validation_indexes])
                final_base = estimator.predict_proba(matrix[test_indexes])
                fit_and_neighbor_seconds = time.perf_counter() - fit_started
                for probability_scale in probability_scales:
                    started = time.perf_counter()
                    validation_probabilities = np.clip(
                        validation_base * probability_scale,
                        0.0,
                        1.0,
                    )
                    thresholds = quality.tune_thresholds(
                        validation_probabilities,
                        truth[validation_indexes],
                        classes,
                    )
                    final_probabilities = np.clip(
                        final_base * probability_scale,
                        0.0,
                        1.0,
                    )
                    evaluation = quality.evaluate(
                        final_probabilities,
                        truth[test_indexes],
                        classes,
                        thresholds,
                    )
                    estimator.probability_scale = probability_scale
                    model = MultiLabelCrateModel(estimator, classes)
                    artifact = (
                        Path(tempfile.mkdtemp())
                        / (
                            f"knn-{metric}-{count}-power-{distance_power:g}"
                            f"-scale-{probability_scale:g}.joblib"
                        )
                    )
                    joblib.dump(
                        {
                            "model": model,
                            "feature_columns": [
                                f"feature_{index}"
                                for index in range(matrix.shape[1])
                            ],
                            "prediction_semantics": "independent_multilabel",
                        },
                        artifact,
                        compress=3,
                    )
                    results.append({
                        "name": (
                            f"shared_knn_{metric}_{count}"
                            f"_power_{distance_power:g}"
                            f"_scale_{probability_scale:g}"
                        ),
                        "metric": metric,
                        "neighbors": count,
                        "distance_power": distance_power,
                        "probability_scale": probability_scale,
                        "fit_and_neighbor_seconds": fit_and_neighbor_seconds,
                        "evaluation_seconds": time.perf_counter() - started,
                        "artifact_bytes": artifact.stat().st_size,
                        "model_memory_bytes": estimator.memory_estimate_bytes(),
                        "micro_f1": evaluation.multi_label_metrics.micro_f1,
                        "macro_f1": evaluation.multi_label_metrics.macro_f1,
                        "top3_hit_rate": evaluation.ranking_metrics.top3_hit_rate,
                        "recall_at_3": evaluation.ranking_metrics.recall_at_3,
                        "calibration_error": evaluation.calibration_metrics.expected_calibration_error,
                    })
    for temperature in nearest_positive_temperatures:
        started = time.perf_counter()
        estimator = SharedNearestPositiveClassifier(
            positive_neighbors=positive_neighbors,
            distance_temperature=temperature,
        ).fit(matrix[train_indexes], truth[train_indexes])
        fit_seconds = time.perf_counter() - started
        validation_probabilities = estimator.predict_proba(matrix[validation_indexes])
        thresholds = quality.tune_thresholds(
            validation_probabilities,
            truth[validation_indexes],
            classes,
        )
        final_probabilities = estimator.predict_proba(matrix[test_indexes])
        evaluation = quality.evaluate(
            final_probabilities,
            truth[test_indexes],
            classes,
            thresholds,
        )
        model = MultiLabelCrateModel(estimator, classes)
        artifact = (
            Path(tempfile.mkdtemp())
            / (
                f"nearest-positive-{positive_neighbors}"
                f"-temperature-{temperature:g}.joblib"
            )
        )
        joblib.dump(
            {
                "model": model,
                "feature_columns": [f"feature_{index}" for index in range(matrix.shape[1])],
                "prediction_semantics": "independent_multilabel",
            },
            artifact,
            compress=3,
        )
        results.append({
            "name": (
                f"shared_nearest_positive_{positive_neighbors}"
                f"_temperature_{temperature:g}"
            ),
            "positive_neighbors": positive_neighbors,
            "distance_temperature": temperature,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - started,
            "artifact_bytes": artifact.stat().st_size,
            "model_memory_bytes": estimator.memory_estimate_bytes(),
            "micro_f1": evaluation.multi_label_metrics.micro_f1,
            "macro_f1": evaluation.multi_label_metrics.macro_f1,
            "top3_hit_rate": evaluation.ranking_metrics.top3_hit_rate,
            "recall_at_3": evaluation.ranking_metrics.recall_at_3,
            "calibration_error": evaluation.calibration_metrics.expected_calibration_error,
        })
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "rows": len(collapsed),
        "classes": len(classes),
        "split": {
            "strategy": split.summary.strategy,
            "random_seed": split.summary.random_seed,
            "train_count": split.summary.train_count,
            "validation_count": split.summary.validation_count,
            "test_count": split.summary.test_count,
            "final_test_available": split.summary.final_test_available,
            "leakage_free": split.summary.leakage_free,
            "warnings": list(split.summary.warnings),
        },
        "peak_python_memory_bytes": peak,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    parser.add_argument("--neighbors", default="1,3,7,15")
    parser.add_argument("--distance-powers", default="1")
    parser.add_argument("--probability-scales", default="1")
    parser.add_argument("--nearest-positive-temperatures", default="")
    parser.add_argument("--positive-neighbors", type=int, default=1)
    parser.add_argument("--metrics", default="euclidean")
    args = parser.parse_args()
    values = tuple(int(value) for value in args.neighbors.split(",") if value)
    powers = tuple(
        float(value) for value in args.distance_powers.split(",") if value
    )
    scales = tuple(
        float(value) for value in args.probability_scales.split(",") if value
    )
    temperatures = tuple(
        float(value)
        for value in args.nearest_positive_temperatures.split(",")
        if value
    )
    metrics = tuple(value for value in args.metrics.split(",") if value)
    print(
        json.dumps(
            compare(
                args.features_csv,
                values,
                powers,
                scales,
                temperatures,
                args.positive_neighbors,
                metrics,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
