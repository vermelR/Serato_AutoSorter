"""Evaluate bounded multi-output trees on the frozen M5 split."""

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from serato_ai.core.models import EvaluationConfiguration
from serato_ai.core.quality_rules import deterministic_multilabel_split
from benchmarks.experimental_estimators import (
    BoundedMultiOutputExtraTreesClassifier,
)
from serato_ai.services.legacy_compact_migration_service import (
    LegacyCompactMigrationService,
)
from serato_ai.services.model_quality_service import ModelQualityService
from serato_ai.services.training_service import MultiLabelCrateModel


def tree_memory_bytes(estimators) -> tuple[int, int]:
    total = 0
    nodes = 0
    for fitted in estimators:
        tree = fitted.tree_
        nodes += tree.node_count
        total += tree.value.nbytes
        total += tree.children_left.nbytes + tree.children_right.nbytes
        total += tree.feature.nbytes + tree.threshold.nbytes
        total += tree.impurity.nbytes + tree.n_node_samples.nbytes
        total += tree.weighted_n_node_samples.nbytes
    return int(total), nodes


def compare(
    features_csv: str | Path,
    configurations: tuple[tuple[int, int, int], ...],
    ovr_configurations: tuple[tuple[int, int, int], ...] = (),
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

    def indexes(values):
        return [group_index[value.track_id] for value in values]

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
    for tree_count, leaf_count, min_leaf in configurations:
        started = time.perf_counter()
        estimator = BoundedMultiOutputExtraTreesClassifier(
            n_estimators=tree_count,
            max_leaf_nodes=leaf_count,
            min_samples_leaf=min_leaf,
            random_state=42,
            worker_count=1,
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
        artifact = Path(tempfile.mkdtemp()) / (
            f"extra-trees-{tree_count}-{leaf_count}-{min_leaf}.joblib"
        )
        joblib.dump(
            {
                "model": model,
                "feature_columns": [
                    f"feature_{index}" for index in range(matrix.shape[1])
                ],
                "prediction_semantics": "independent_multilabel",
            },
            artifact,
            compress=3,
        )
        results.append(
            {
                "name": (
                    f"bounded_extra_trees_{tree_count}_leaves_{leaf_count}"
                    f"_min_leaf_{min_leaf}"
                ),
                "fit_seconds": fit_seconds,
                "total_seconds": time.perf_counter() - started,
                "artifact_bytes": artifact.stat().st_size,
                "model_memory_bytes": estimator.memory_estimate_bytes(),
                "node_count": estimator.node_count(),
                "micro_f1": evaluation.multi_label_metrics.micro_f1,
                "macro_f1": evaluation.multi_label_metrics.macro_f1,
                "top3_hit_rate": evaluation.ranking_metrics.top3_hit_rate,
                "recall_at_3": evaluation.ranking_metrics.recall_at_3,
                "calibration_error": (
                    evaluation.calibration_metrics.expected_calibration_error
                ),
            }
        )
    for tree_count, leaf_count, min_leaf in ovr_configurations:
        started = time.perf_counter()
        estimator = OneVsRestClassifier(
            ExtraTreesClassifier(
                n_estimators=tree_count,
                max_leaf_nodes=leaf_count,
                min_samples_leaf=min_leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            n_jobs=1,
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
        artifact = Path(tempfile.mkdtemp()) / (
            f"ovr-extra-trees-{tree_count}-{leaf_count}-{min_leaf}.joblib"
        )
        joblib.dump(
            {
                "model": model,
                "feature_columns": [
                    f"feature_{index}" for index in range(matrix.shape[1])
                ],
                "prediction_semantics": "independent_multilabel",
            },
            artifact,
            compress=3,
        )
        forests = [
            tree
            for classifier in estimator.estimators_
            for tree in classifier.estimators_
        ]
        memory_bytes, node_count = tree_memory_bytes(forests)
        results.append(
            {
                "name": (
                    f"ovr_extra_trees_{tree_count}_leaves_{leaf_count}"
                    f"_min_leaf_{min_leaf}"
                ),
                "fit_seconds": fit_seconds,
                "total_seconds": time.perf_counter() - started,
                "artifact_bytes": artifact.stat().st_size,
                "model_memory_bytes": memory_bytes,
                "node_count": node_count,
                "micro_f1": evaluation.multi_label_metrics.micro_f1,
                "macro_f1": evaluation.multi_label_metrics.macro_f1,
                "top3_hit_rate": evaluation.ranking_metrics.top3_hit_rate,
                "recall_at_3": evaluation.ranking_metrics.recall_at_3,
                "calibration_error": (
                    evaluation.calibration_metrics.expected_calibration_error
                ),
            }
        )
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
    parser.add_argument(
        "--configurations",
        default="32:64:1,48:128:1",
        help="Comma-separated trees:max-leaf-nodes:min-samples-leaf values.",
    )
    parser.add_argument(
        "--ovr-configurations",
        default="",
        help="Optional one-vs-rest trees:max-leaf-nodes:min-samples-leaf values.",
    )
    args = parser.parse_args()
    values = tuple(
        tuple(int(part) for part in value.split(":"))
        for value in args.configurations.split(",")
        if value
    )
    if any(len(value) != 3 for value in values):
        parser.error("Each configuration must contain trees:leaves:min-leaf.")
    ovr_values = tuple(
        tuple(int(part) for part in value.split(":"))
        for value in args.ovr_configurations.split(",")
        if value
    )
    if any(len(value) != 3 for value in ovr_values):
        parser.error("Each OVR configuration must contain trees:leaves:min-leaf.")
    print(json.dumps(compare(args.features_csv, values, ovr_values), indent=2))


if __name__ == "__main__":
    main()
