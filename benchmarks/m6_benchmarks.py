"""Repeatable local-only Milestone 6 scalability benchmarks.

The benchmark uses generated scalar features, empty temporary audio fixtures,
temporary Serato roots, and local stores. It never reads a real Serato library,
modifies source music, or uses the network.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from serato_ai.core.dataframes import paginate_prediction_frame
from serato_ai.core.models import EvaluationConfiguration
from serato_ai.infrastructure.application_data import ApplicationDataPaths
from serato_ai.infrastructure.feature_cache_store import CachedFeatures, FeatureCacheStore
from serato_ai.infrastructure.local_json import write_json_atomic
from serato_ai.infrastructure.personal_model_store import PersonalModelStore
from serato_ai.infrastructure.queue_store import QueueStore
from serato_ai.infrastructure.scan_index_store import ScanIndexStore
from serato_ai.services.compact_estimators import SharedKNNMultiLabelClassifier
from serato_ai.services.library_scan_service import LibraryScanService
from serato_ai.services.model_quality_service import ModelQualityService
from serato_ai.services.model_service import ModelService, clear_model_cache
from serato_ai.services.training_service import MultiLabelCrateModel
from serato_crate import serialize_serato_track_record


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    tracks: int
    crates: int


PROFILES = {
    "small": BenchmarkProfile("small", 500, 20),
    "medium": BenchmarkProfile("medium", 10_000, 100),
    "large": BenchmarkProfile("large", 50_000, 300),
}


def _seconds(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _startup_timings(data_root: Path) -> tuple[float, float]:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["SERATO_AI_DATA_DIR"] = str(data_root)
    command = (
        "import importlib,json,time;"
        "started=time.perf_counter();"
        "import app;"
        "cold=time.perf_counter()-started;"
        "started=time.perf_counter();"
        "importlib.import_module('app');"
        "warm=time.perf_counter()-started;"
        "print(json.dumps([cold,warm]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    cold, warm = json.loads(result.stdout.strip().splitlines()[-1])
    return float(cold), float(warm)


def _synthetic_matrices(profile: BenchmarkProfile):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(profile.tracks, 16)).astype(np.float32)
    truth = np.zeros((profile.tracks, profile.crates), dtype=np.uint8)
    indexes = np.arange(profile.tracks)
    truth[indexes, indexes % profile.crates] = 1
    secondary = indexes % 5 == 0
    truth[indexes[secondary], (indexes[secondary] + 7) % profile.crates] = 1
    return features, truth


def _materialize_scan_fixture(
    root: Path,
    profile: BenchmarkProfile,
) -> tuple[Path, Path, Path, Path]:
    serato_root = root / "synthetic-serato"
    subcrates = serato_root / "SubCrates"
    music = root / "synthetic-music"
    subcrates.mkdir(parents=True)
    music.mkdir()
    first_track: Path | None = None
    bucket_count = min(100, max(1, profile.tracks // 500))
    buckets = [music / f"bucket-{index:03d}" for index in range(bucket_count)]
    for bucket in buckets:
        bucket.mkdir()
    crate_payloads = [
        bytearray(b"vrsn\x00\x00\x00\x00")
        for _ in range(profile.crates)
    ]
    for index in range(profile.tracks):
        track = buckets[index % bucket_count] / f"track-{index:06d}.mp3"
        track.touch()
        first_track = first_track or track
        crate_payloads[index % profile.crates].extend(
            serialize_serato_track_record(str(track.resolve()).lstrip("/"))
        )
    first_crate: Path | None = None
    for index, payload in enumerate(crate_payloads):
        crate = subcrates / f"Crate-{index:04d}.crate"
        crate.write_bytes(bytes(payload))
        first_crate = first_crate or crate
    assert first_track is not None and first_crate is not None
    return serato_root, music, first_track, first_crate


def _scan_benchmarks(
    root: Path,
    paths: ApplicationDataPaths,
    profile: BenchmarkProfile,
) -> dict:
    serato_root, music, first_track, first_crate = _materialize_scan_fixture(root, profile)
    scanner = LibraryScanService(
        scan_index=ScanIndexStore(paths, paths.root / "benchmark-scan-index.json")
    )
    full, full_seconds = _seconds(
        lambda: scanner.scan(serato_root, (str(music),), full_rescan=True)
    )
    full_summary = full.scan_summary
    del full
    gc.collect()
    unchanged, unchanged_seconds = _seconds(
        lambda: scanner.scan(serato_root, (str(music),))
    )
    unchanged_summary = unchanged.scan_summary
    del unchanged
    gc.collect()
    first_track.write_bytes(b"modified")
    os.utime(first_crate, None)
    changed, changed_seconds = _seconds(
        lambda: scanner.scan(serato_root, (str(music),))
    )
    changed_summary = changed.scan_summary
    changed_fingerprint = changed.library_fingerprint
    del changed
    gc.collect()
    clean = LibraryScanService(
        scan_index=ScanIndexStore(paths, paths.root / "benchmark-clean-index.json")
    )
    clean_result, verification_seconds = _seconds(
        lambda: clean.scan(serato_root, (str(music),), full_rescan=True)
    )
    clean_fingerprint = clean_result.library_fingerprint
    del clean_result
    gc.collect()
    return {
        "full_seconds": full_seconds,
        "unchanged_incremental_seconds": unchanged_seconds,
        "changed_incremental_seconds": changed_seconds,
        "clean_verification_full_seconds": verification_seconds,
        "full_crates_parsed": full_summary.crate_files_parsed,
        "full_music_file_stat_checks": full_summary.file_stat_checks,
        "full_directories_enumerated": full_summary.directories_scanned,
        "unchanged_crates_parsed": unchanged_summary.crate_files_parsed,
        "unchanged_crates_reused": unchanged_summary.crate_files_reused,
        "unchanged_music_files_reused": unchanged_summary.music_files_reused,
        "unchanged_music_file_stat_checks": unchanged_summary.file_stat_checks,
        "unchanged_directories_enumerated": unchanged_summary.directories_scanned,
        "changed_crates_parsed": changed_summary.crate_files_parsed,
        "changed_directories_enumerated": changed_summary.directories_scanned,
        "changed_music_items": changed_summary.changed_item_count,
        "incremental_equals_clean_full": (
            changed_fingerprint == clean_fingerprint
        ),
        "music_files_seen": changed_summary.music_files_seen,
    }


def run_profile(profile: BenchmarkProfile, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    paths = ApplicationDataPaths(root / "application-data")
    paths.ensure()
    rss_before = _rss_bytes()
    tracemalloc.start()
    cold_startup, warm_startup = _startup_timings(paths.root / "startup")

    (features, truth), preparation_seconds = _seconds(
        lambda: _synthetic_matrices(profile)
    )
    split_at = max(1, int(profile.tracks * 0.8))
    train_features = features[:split_at]
    train_truth = truth[:split_at]
    evaluate_count = min(1_000, profile.tracks - split_at)
    if evaluate_count <= 0:
        evaluate_count = min(1_000, profile.tracks)
        evaluation_features = features[:evaluate_count]
        evaluation_truth = truth[:evaluate_count]
    else:
        evaluation_features = features[split_at : split_at + evaluate_count]
        evaluation_truth = truth[split_at : split_at + evaluate_count]

    estimator = SharedKNNMultiLabelClassifier(
        n_neighbors=35,
        metric="cosine",
        probability_scale=1.75,
    )
    _, candidate_training_seconds = _seconds(
        lambda: estimator.fit(train_features, train_truth)
    )
    classes = tuple(f"Crate%%{index:04d}" for index in range(profile.crates))
    model = MultiLabelCrateModel(estimator, classes)
    bundle = {
        "model": model,
        "feature_columns": [f"feature_{index}" for index in range(16)],
        "prediction_semantics": "independent_multilabel",
        "bundle_schema_version": "seratoai-inference-v1",
        "model_version": f"benchmark-{profile.name}",
    }
    artifact = root / "benchmark-model.joblib"
    _, serialization_seconds = _seconds(
        lambda: joblib.dump(bundle, artifact, compress=3)
    )
    models = PersonalModelStore(paths)
    loader = lambda value: joblib.load(value)
    service = ModelService(loader=loader, personal_store=models)
    clear_model_cache()
    _, cold_model_load_seconds = _seconds(lambda: service.load(str(artifact)))
    _, warm_model_load_seconds = _seconds(lambda: service.load(str(artifact)))
    _, single_prediction_seconds = _seconds(
        lambda: model.predict_proba(evaluation_features[:1])
    )
    prediction_batch = evaluation_features[: min(128, len(evaluation_features))]
    _, batch_prediction_seconds = _seconds(
        lambda: model.predict_proba(prediction_batch)
    )

    quality = ModelQualityService(EvaluationConfiguration(random_seed=42))

    def evaluate_candidate():
        probabilities = model.predict_proba(evaluation_features)
        thresholds = quality.tune_thresholds(
            probabilities,
            evaluation_truth,
            classes,
        )
        return quality.evaluate(
            probabilities,
            evaluation_truth,
            classes,
            thresholds,
        )

    evaluation, evaluation_seconds = _seconds(evaluate_candidate)

    cache = FeatureCacheStore(paths)
    cache_rows = [
        CachedFeatures(
            f"fingerprint-{index:06d}",
            "audio-v1-16",
            tuple(float(value) for value in features[index]),
        )
        for index in range(profile.tracks)
    ]
    _, cache_write_seconds = _seconds(lambda: cache.put_many(cache_rows))
    del cache_rows
    lookup_count = min(2_000, profile.tracks)
    _, cache_lookup_seconds = _seconds(
        lambda: [
            cache.get(f"fingerprint-{index:06d}", "audio-v1-16")
            for index in range(lookup_count)
        ]
    )

    queue = QueueStore(
        root / "pending.sqlite3",
        root / "processed.json",
        maximum_pending=10_000,
    )
    burst = [
        {
            "path": f"/synthetic/track-{index:06d}.mp3",
            "_file_identity": f"synthetic:{index}",
        }
        for index in range(profile.tracks)
    ]
    accepted, watcher_burst_seconds = _seconds(lambda: queue.append_many(burst))
    accepted_count = sum(accepted)
    del accepted, burst

    def prepare_table():
        frame = pd.DataFrame(
            {
                "path": [f"/synthetic/track-{index:06d}.mp3" for index in range(profile.tracks)],
                "Song Title": [f"Track {index}" for index in range(profile.tracks)],
                "Artist": "Synthetic",
                "Genre": "",
                "Year": "",
                "Top Suggested Crate": [
                    classes[index % profile.crates]
                    for index in range(profile.tracks)
                ],
                "Approve": (np.arange(profile.tracks) % 11 == 0),
                "Needs Review": (np.arange(profile.tracks) % 3 == 0),
            }
        )
        page = paginate_prediction_frame(
            frame,
            search="Track",
            view="Low confidence only",
            page=2,
            page_size=50,
        )
        return frame, page

    (table, table_page), table_preparation_seconds = _seconds(prepare_table)
    scan = _scan_benchmarks(root / "scan-fixture", paths, profile)
    python_current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    estimated_live_bytes = int(
        features.nbytes
        + truth.nbytes
        + estimator.memory_estimate_bytes()
        + table.memory_usage(index=True, deep=True).sum()
    )
    return {
        "profile": asdict(profile),
        "recorded_at": datetime.now(UTC).isoformat(),
        "startup": {
            "cold_seconds": cold_startup,
            "warm_seconds": warm_startup,
        },
        "model": {
            "training_preparation_seconds": preparation_seconds,
            "candidate_training_seconds": candidate_training_seconds,
            "serialization_seconds": serialization_seconds,
            "cold_load_seconds": cold_model_load_seconds,
            "warm_load_seconds": warm_model_load_seconds,
            "disk_load_calls": service.disk_load_count,
            "artifact_bytes": artifact.stat().st_size,
            "in_memory_bytes": estimator.memory_estimate_bytes(),
        },
        "prediction": {
            "single_seconds": single_prediction_seconds,
            "batch_seconds": batch_prediction_seconds,
            "batch_items": len(prediction_batch),
            "estimator_calls_for_batch": 1,
        },
        "evaluation": {
            "items": len(evaluation_features),
            "seconds": evaluation_seconds,
            "micro_f1": evaluation.multi_label_metrics.micro_f1,
            "top3_hit_rate": evaluation.ranking_metrics.top3_hit_rate,
        },
        "feature_cache": {
            "write_items": profile.tracks,
            "write_seconds": cache_write_seconds,
            "lookup_items": lookup_count,
            "lookup_seconds": cache_lookup_seconds,
            "hits": cache.stats().hits,
            "misses": cache.stats().misses,
            "bytes": cache.size_bytes(),
        },
        "watcher_burst": {
            "events": profile.tracks,
            "seconds": watcher_burst_seconds,
            "accepted": accepted_count,
            "bounded_capacity": queue.maximum_pending,
            "stored": queue.count_pending(),
        },
        "scanning": scan,
        "table": {
            "records": len(table),
            "preparation_seconds": table_preparation_seconds,
            "rendered_rows": len(table_page.frame),
            "filtered_rows": table_page.filtered_rows,
            "page_size": table_page.page_size,
        },
        "memory": {
            "estimated_live_bytes": estimated_live_bytes,
            "python_current_bytes": python_current,
            "python_peak_bytes": python_peak,
            "process_peak_rss_bytes": max(rss_before, _rss_bytes()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        default="small,medium,large",
        help="Comma-separated profile names: small, medium, large.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    unknown = sorted(set(selected) - set(PROFILES))
    if unknown:
        parser.error("Unknown benchmark profile(s): " + ", ".join(unknown))
    results = []
    with tempfile.TemporaryDirectory(prefix="seratoai-m6-benchmarks-") as temporary:
        base = Path(temporary)
        for name in selected:
            results.append(run_profile(PROFILES[name], base / name))
    report = {
        "schema_version": "seratoai-m6-benchmarks-v1",
        "safety": (
            "Synthetic scalar features, temporary empty audio fixtures, temporary "
            "Serato roots, no source-audio writes, and no network access."
        ),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, report)
    print(text)


if __name__ == "__main__":
    main()
