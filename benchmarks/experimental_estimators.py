"""Exploratory estimators used only by Milestone 6 comparison benchmarks."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier


class BoundedMultiOutputExtraTreesClassifier:
    """Bound tree count and leaves to prevent legacy node-value explosion."""

    def __init__(
        self,
        *,
        n_estimators: int = 48,
        max_leaf_nodes: int = 128,
        min_samples_leaf: int = 1,
        max_features: str | float | int | None = "sqrt",
        random_state: int = 42,
        worker_count: int = 1,
    ):
        self.n_estimators = max(1, min(128, int(n_estimators)))
        self.max_leaf_nodes = max(2, min(512, int(max_leaf_nodes)))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.max_features = max_features
        self.random_state = int(random_state)
        self.worker_count = max(1, min(4, int(worker_count)))

    def fit(self, features, truth):
        matrix = np.asarray(features, dtype=np.float32)
        labels = np.asarray(truth, dtype=np.uint8)
        if (
            matrix.ndim != 2
            or labels.ndim != 2
            or matrix.shape[0] != labels.shape[0]
        ):
            raise ValueError(
                "Bounded Extra Trees requires aligned 2D feature and label matrices."
            )
        self.output_count_ = labels.shape[1]
        self.estimator_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.worker_count,
        )
        self.estimator_.fit(matrix, labels)
        return self

    def predict_proba(self, features):
        if not hasattr(self, "estimator_"):
            raise ValueError("Bounded Extra Trees has not been fitted.")
        rows = len(features)
        output = np.empty((rows, self.output_count_), dtype=np.float32)
        probabilities = self.estimator_.predict_proba(
            np.asarray(features, dtype=np.float32)
        )
        for index, values in enumerate(probabilities):
            classes = np.asarray(self.estimator_.classes_[index])
            positive = np.flatnonzero(classes == 1)
            if len(positive):
                output[:, index] = np.asarray(values)[:, int(positive[0])]
            else:
                output[:, index] = (
                    1.0 if len(classes) and classes[0] == 1 else 0.0
                )
        return output

    def node_count(self) -> int:
        return sum(
            tree.tree_.node_count for tree in self.estimator_.estimators_
        )

    def memory_estimate_bytes(self) -> int:
        total = 0
        for tree in self.estimator_.estimators_:
            state = tree.tree_
            total += state.value.nbytes
            total += state.children_left.nbytes + state.children_right.nbytes
            total += state.feature.nbytes + state.threshold.nbytes
            total += state.impurity.nbytes + state.n_node_samples.nbytes
            total += state.weighted_n_node_samples.nbytes
        return int(total)


class SharedNearestPositiveClassifier:
    """Exploratory one-shared-matrix nearest-positive scorer."""

    def __init__(
        self,
        *,
        positive_neighbors: int = 1,
        distance_temperature: float = 1.0,
        prediction_batch_size: int = 32,
    ):
        self.positive_neighbors = max(1, min(16, int(positive_neighbors)))
        self.distance_temperature = max(0.01, float(distance_temperature))
        self.prediction_batch_size = max(
            1,
            min(128, int(prediction_batch_size)),
        )

    def fit(self, features, truth):
        matrix = np.asarray(features, dtype=np.float32)
        labels = np.asarray(truth, dtype=np.uint8)
        if (
            matrix.ndim != 2
            or labels.ndim != 2
            or matrix.shape[0] != labels.shape[0]
        ):
            raise ValueError(
                "Nearest-positive scoring requires aligned 2D matrices."
            )
        self.feature_mean_ = np.mean(
            matrix,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        scale = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale <= 1e-7)] = 1.0
        self.feature_scale_ = scale
        self.training_features_ = np.ascontiguousarray(
            (matrix - self.feature_mean_) / self.feature_scale_,
            dtype=np.float32,
        )
        self.training_squared_norms_ = np.sum(
            self.training_features_ * self.training_features_,
            axis=1,
            dtype=np.float32,
        )
        self.training_labels_ = sparse.csc_matrix(labels, dtype=np.uint8)
        self.output_count_ = labels.shape[1]
        return self

    def predict_proba(self, features):
        if not hasattr(self, "training_features_"):
            raise ValueError("Nearest-positive classifier has not been fitted.")
        matrix = np.asarray(features, dtype=np.float32)
        normalized = np.ascontiguousarray(
            (matrix - self.feature_mean_) / self.feature_scale_,
            dtype=np.float32,
        )
        output = np.zeros(
            (len(normalized), self.output_count_),
            dtype=np.float32,
        )
        label_index = self.training_labels_
        for start in range(0, len(normalized), self.prediction_batch_size):
            batch = normalized[start : start + self.prediction_batch_size]
            query_norms = np.sum(batch * batch, axis=1, keepdims=True)
            squared = (
                query_norms
                + self.training_squared_norms_[None, :]
                - 2.0 * batch @ self.training_features_.T
            )
            np.maximum(squared, 0.0, out=squared)
            distances = np.sqrt(squared, out=squared)
            for output_index in range(self.output_count_):
                left = label_index.indptr[output_index]
                right = label_index.indptr[output_index + 1]
                positive_indexes = label_index.indices[left:right]
                if not len(positive_indexes):
                    continue
                positive_distances = distances[:, positive_indexes]
                count = min(
                    self.positive_neighbors,
                    positive_distances.shape[1],
                )
                if count == 1:
                    local_distance = np.min(positive_distances, axis=1)
                else:
                    nearest = np.partition(
                        positive_distances,
                        count - 1,
                        axis=1,
                    )[:, :count]
                    local_distance = np.mean(nearest, axis=1)
                output[
                    start : start + len(batch),
                    output_index,
                ] = np.exp(-local_distance / self.distance_temperature)
        return output

    def memory_estimate_bytes(self) -> int:
        labels = self.training_labels_
        return int(
            self.training_features_.nbytes
            + self.training_squared_norms_.nbytes
            + self.feature_mean_.nbytes
            + self.feature_scale_.nbytes
            + labels.data.nbytes
            + labels.indices.nbytes
            + labels.indptr.nbytes
        )
