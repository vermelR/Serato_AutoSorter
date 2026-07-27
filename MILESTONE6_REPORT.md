# Milestone 6 completion report

This report is evidence for the acceptance checklist in the Milestone 6
specification. All production-library claims use read-only inspection. Model
migration and UI smoke work use application data inside this repository,
synthetic/empty fixtures, and offline providers.

## 1. Baseline tests, coverage, performance, and storage

Before Milestone 6 changes:

- Full suite: 174 passed.
- Writer suite: 47 passed.
- Streamlit suite: 11 passed.
- Fast suite: 126 passed.
- Model-quality suite: 11 passed.
- Coverage: 82.97%.
- Managed active/previous/candidate model and application cache/feedback
  stores: absent in the real per-user application-data location.
- Workspace legacy model: 33,878,876,869 bytes (31.55 GiB).
- Saved scalar feature source: 17,985,351 bytes; 38,317 references, 8,159
  grouped physical/path identities, 919 source crates, 827 eligible classes.

The legacy model was deliberately not loaded for baseline timing or prediction:
its 31.55 GiB file size and source object graph made an ad hoc deserialize an
unsafe memory experiment. Baseline scans had no persistent signature index,
model loading had no immutable-version cache, prediction invoked inference
per track, the JSON feature cache required whole-file access, and the watcher
queue was rebuilt from JSONL. The repeatable final benchmark below records the
measurable replacement behavior.

## 2. Exact 40 GB root cause

`phase3_train.py` fits one multiclass `RandomForestClassifier` with 400
unbounded trees and 919 string classes. Every sklearn tree stores
`tree_.value` as a dense node × class-count array. The resulting growth is
approximately trees × nodes × classes, not just trees × nodes.

Disposable reproductions proved the object graph without loading the legacy
file:

| Fixture | Artifact | `tree_.value` |
| --- | ---: | ---: |
| 2,000 rows, 100 classes, 10 trees | 19.28 MiB | 17.84 MiB |
| 4,000 rows, 200 classes, 10 trees | 77.49 MiB | 74.49 MiB |

The second fixture spends about 96% of tree-array bytes on dense class values.
Scaling the same structure to 919 classes and 400 trees explains the
33,878,876,869-byte artifact. Source and header inspection found no embedded
raw audio, feature cache, validation/test DataFrame, feedback history, or
candidate history in that legacy bundle.

## 3. Model-size breakdown

The validated active compact artifact is 636,296 bytes:

- One normalized `float32` training-feature matrix shared by all outputs.
- One sparse CSR multi-label matrix.
- One nearest-neighbor index referencing the shared feature data.
- Raw class names (827 eligible crates).
- Feature schema, thresholds, compact quality configuration, bundle/version
  identity, compression/storage status, and exact size metadata.

External training-only storage for the same validation run:

- Preparation matrices: 558,244 bytes.
- Preparation metadata: 629,844 bytes.
- Two completed candidate-evaluation cache artifacts plus metadata: about
  11.7 MiB.
- Full final evaluation records: 819,383 bytes compressed.
- Bounded quality summary history: 2,003,869 bytes.
- Full external model/training metadata: about 13 MiB.

The active artifact is about 53,244 times smaller than the legacy model
(99.9981% reduction).

## 4. Files added and changed

The implementation adds compact/storage/cache/index/diagnostic infrastructure,
background jobs, the migration service, benchmark programs/results, M6 tests,
and this report. Existing model, prediction, scan, watcher, onboarding,
health, Streamlit, settings, and documentation modules were extended. The
legacy `serato_model.pkl`, saved feature CSV, Serato crates, and source audio
were not edited.

## 5. Final model-bundle and external-store design

`seratoai-inference-v1` contains only inference-required state. Known
training-only keys are stripped before a compressed atomic write. Full
`ModelVersionMetadata`, preparation matrices, completed candidate evaluations,
feature vectors, feedback, quality summaries, full evaluation rows, scan
index, job checkpoints, performance records, and prior/rejected models live in
separate local stores.

Active and previous manifests point to immutable files under
`models/versions`; unactivated artifacts live under `models/candidates`.
Artifact size is measured through a self-consistent serialization pass and is
identical in the filesystem, bundle top level, compact inference metadata, and
external metadata. A bounded deterministic fixpoint nonce handles the rare
case where changing the embedded integer changes zlib output by one byte; a
16-layout regression proves the stored and physical sizes remain exact.

## 6. Compact model comparison

All comparisons use the deterministic frozen physical-track split.

| Design | Result |
| --- | --- |
| Legacy 400-tree multiclass forest | 33.88 GB; pathological dense class values; not safe to load |
| Compact balanced one-vs-rest logistic | Compact, but failed unchanged gate (Micro F1 0.0101; Top-3 0.0141) |
| Bounded multi-output Extra Trees | 2.37–6.27 MB in explored bounds; Top-3 up to about 0.276; failed gate |
| Shared nearest-neighbor, k=1 | 622 KB-class artifact; Micro F1 0.0583; Top-3 0.1268 |
| Shared nearest-neighbor, cosine k=35, scale 1.75 | 636,296 bytes; passed unchanged gate |

Exploratory estimators live under `benchmarks/`; only the selected shared KNN
is in the production services package.

## 7. Quality comparison

The normal `TrainingService`/`ModelActivationService` migration selected
`SharedKNNMultiLabelClassifier` on validation data and evaluated it once on
the final split:

- Micro F1: 0.1016399695.
- Macro F1: 0.0241673984.
- Top-3 hit rate: 0.3026960784.
- Recall@3: 0.1942735186.
- Expected calibration error: 0.0067113859.
- Classes preserved: all 827 eligible crates.
- Final evaluation records: 1,632.
- Unchanged Milestone 5 gate: passed against the deterministic first-model
  baseline.

The failed logistic candidate was not activated.

## 8. Final model and cache sizes

The isolated final migration root is `.m6-final-validation-v2`:

- Active model: 636,296 bytes under `models/versions`.
- Candidate directory after activation: empty.
- Feature cache: empty for CSV migration (the saved scalar training cache is
  used instead).
- Training cache: about 13 MiB, including two reusable candidate evaluations.
- Evaluation records: 819,383 bytes.
- Quality history: 2,003,869 bytes.
- Legacy reference: points to the unchanged 33,878,876,869-byte source.

## 9. Size-gate behavior

Defaults are 500 MiB preferred, 2 GiB warning, 5 GiB automatic-activation
review block, and 8 GiB hard block. Tests cover compact, preferred-exceeded,
warning, review, hard-block, and deliberate developer override states. An
oversized serialized candidate remains in candidate storage for review while
active/previous pointers remain unchanged.

## 10. Migration and rollback behavior

The final isolated migration:

- read `music_features_dataset.csv`;
- only called `stat` on the oversized pickle;
- grouped duplicate crate rows into multi-label physical/path identities;
- trained, evaluated, saved, measured, reloaded, smoke-predicted, and gated a
  compact candidate;
- atomically moved it to immutable active version storage;
- preserved the legacy file byte size and recorded a recoverable reference.

Activation retains one previous known-good version. Restore swaps manifests
atomically. A corrupt active artifact falls back read-only to the prior model
without rewriting either pointer.

## 11. Model-loading optimization

The process-wide locked cache keys immutable model path, mtime, size, and
loader identity. Repeated loads of one version perform one disk load; a file
or version change reloads; obsolete cache entries are released. The final
compact validation confirmed two service loads with one disk load.

## 12. Batch-prediction result

Audio extraction is performed first, then model calls run in bounded batches
(128 by default) with per-track failure isolation/fallback. Regression tests
prove batch and individual calls have the same raw rankings, probabilities,
confidence, thresholds, review reasons, quality labels, allow-list behavior,
and manual/watcher schema while reducing four estimator calls to one for four
tracks.

## 13. Incremental-scan result

Crate signatures reuse unchanged parses and retain the last valid record for a
temporarily malformed update. Music path/size/mtime/device/inode and directory
signatures detect add, modify, inode-preserving move, delete, disconnect, and
reconnect. Unchanged scans enumerate zero music directories in all benchmark
profiles. Incremental fingerprints and match results equal a clean full scan.
Cancellation leaves the prior index byte-for-byte intact.

## 14. Watcher-deduplication result

The watcher rejects unsupported, hidden temporary/partial, unreadable,
unstable, already pending, and already processed files before analysis.
Path plus device/inode identity collapses duplicate and rename events. Crate
mtime/size caching avoids unchanged parses and preserves a prior valid view
during a partial update. Transient drive errors use bounded backoff.
When a pending file is renamed, the database row is relocated by device/inode
identity instead of creating a second prediction.

Queue bursts use one SQLite transaction, unique row keys, and a 10,000-row
capacity. Search, category filtering, pending/low-confidence/approved views,
counts, and pages run in SQLite; only the current page enters Streamlit state.

## 15. Background-job and cancellation behavior

`BackgroundJobManager` allows at most two workers, persists the latest 100
snapshots, and reports stage, completed/total, percentage, elapsed time, ETA,
warnings, diagnostics, checkpoint, cancellation, and resume information.
Tests cover completion, failure, cancellation, checkpoint resume, history
bounds, and interrupted-process recovery. Scan/training cancellation occurs
before activation, so incomplete candidates never become active.

## 16. Parallel-extraction result

Low Resource, Balanced, and Fast modes use 1, 2, and at most 4 workers.
Scheduling holds at most the worker count of futures, stops submitting after
cancellation, preserves deterministic input order, isolates a failing track,
and commits each completed cache entry. Sequential and parallel feature
vectors match; a follow-up extraction reuses every completed entry without
calling the extractor.

## 17. Memory improvements

The implementation uses shared `float32` model features, sparse labels,
`uint8` truth, external compressed evaluation rows, indexed SQLite stores,
bounded prediction/extraction batches, current-page queue state, bounded job/
quality/performance history, no whole-cache load, no duplicate model instance,
and explicit release/GC of losing candidates and scan benchmark results.
The 50,000/300 synthetic profile estimates 26.6 MB of live model/data/table
state; process RSS is also reported as a high-water diagnostic.

## 18. Benchmark results

`benchmarks/m6_benchmarks.py` records all required measurements in
`benchmarks/m6-results.json`. The final post-fix run completed on 2026-07-24:

| Profile | Cold startup | Cold model load | Batch prediction | Full / unchanged scan | Watcher burst | Table prep | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 tracks / 20 crates | 0.974 s | 0.00175 s | 100 in 0.0155 s | 0.255 / 0.225 s | 500 in 0.0118 s | 0.0097 s | 198.1 MiB |
| 10,000 / 100 | 0.961 s | 0.00436 s | 128 in 0.0260 s | 5.006 / 4.410 s | 10,000 in 0.215 s | 0.0690 s | 326.7 MiB |
| 50,000 / 300 | 0.965 s | 0.01499 s | 128 in 0.0565 s | 25.679 / 22.566 s | 50,000 in 0.710 s | 0.3365 s | 792.5 MiB |

The large watcher burst accepts/stores the configured 10,000-row maximum.
The full JSON also records feature-cache writes/lookups, evaluation, candidate
training, artifact/in-memory sizes, and calculated live state. Work-count
invariants are:

- one disk model load for cold + warm use;
- one estimator call for a 128-item prediction batch;
- all requested feature lookups hit after one batch write;
- watcher storage never exceeds 10,000 rows;
- unchanged crate parses are zero and all crates are reused;
- unchanged music-directory enumerations are zero;
- incremental fingerprint equals clean full fingerprint;
- table render data is limited to 50 rows at 500, 10,000, and 50,000 records.

## 19. Model Health changes

Model Health displays active, previous, and largest inactive candidate size;
feature/training cache, quality/evaluation/benchmark/feedback storage;
reclaimable bytes; model load and batch latency; cache hit rate; last
incremental scan; model-size and legacy warnings. Confirmed actions clean
rejected candidates, interrupted temporaries, obsolete reports, and obsolete
model versions, or compact the feature cache in a background job. Active,
previous, feedback, live cache, recent quality, and legacy artifacts are
protected by default.

## 20. Tests added

M6 tests cover root-cause scaling, version/compression/size metadata,
training-only exclusion, external evaluation records, load caching/fallback,
size blocks, retention safety, batch equivalence, feature-cache behavior,
candidate-evaluation reuse/invalidation, complete incremental-scan mutations,
external-drive reconnect, cancellation, SQLite queue paging and bursts,
background failure/ETA/resume/history, parallel failure/cancellation/cache
retention, scalable tables, Streamlit no-expensive-rerun instrumentation,
legacy migration, and repeatable benchmark work counts.

## 21. Final tests and coverage

The final exact command matrix is clean:

- Public entry-point compilation and package-wide `compileall`: passed.
- Full suite: 205 passed.
- Writer suite: 47 passed.
- Streamlit AppTest suite: 13 passed.
- Fast suite (`not slow and not integration`): 156 passed.
- Focused Milestone 6 suite: 27 passed.
- Combined performance/storage/size/cache/scan/watcher/job/batch/migration
  markers: 57 passed.
- Import smoke tests: passed.
- Coverage suite: 205 passed at 82.80%.
- Small, medium, and large benchmark profiles: completed.

Coverage is within 0.17 percentage point of the 82.97% pre-change baseline
despite the substantial new storage, scanning, queue, and background-job
surface. Serialization, model gates, caching, scans, watcher behavior, jobs,
batching, cleanup, and migration all have direct behavioral tests.

## 22. Manual smoke-test result

Streamlit is running at `http://127.0.0.1:8501` and was opened in Safari.
The live endpoint returned HTTP 200. A disposable-root AppTest rendered Model
Health without exceptions and showed the active compact model, 621.4 KiB
artifact size, passed quality gate, SharedKNN estimator, legacy 31.6 GiB
warning, storage/performance sections, and a 0.003-second model load.

A real two-second disposable WAV was predicted successfully without online
lookup; manual Final Crates retained raw Serato names. Folder-watcher and
crate-watcher ingestion of that same file converged to one queued row, and the
source SHA remained unchanged. A 120-row queue smoke rendered only 50 rows and
preserved search, pagination, approval, and multi-crate edits without model
load, scan, extraction, prediction, evaluation, or training calls. The normal
migration path trained/evaluated/activated the compact candidate, and focused
smoke tests exercised progress, cancellation, size blocking, prior-version
restore, and corrupt-active fallback using disposable stores.

## 23. No real crate or source audio modification

No command points at the installed Serato library. Tests enforce temporary
Serato roots and block production-root filesystem access. Migration only reads
the saved scalar CSV and stats the legacy pickle. Benchmarks use temporary
empty audio placeholders. Writer tests operate on disposable crate copies.
The legacy model and saved feature CSV retain their original sizes.

## 24. Remaining technical debt

- Music incremental scans avoid unchanged directory enumeration but still
  stat known files to detect in-place content changes that do not alter a
  directory mtime.
- Shared KNN necessarily retains its compact shared inference reference matrix;
  future libraries far beyond the benchmark may benefit from a memory-mapped
  or approximate-neighbor index, subject to the same frozen quality gate.
- Full external training metadata intentionally remains verbose for local
  audit/reproducibility; it is separate from the active artifact.
- Process peak RSS is a high-water measurement and can include earlier phases;
  the benchmark therefore records both process high-water and calculated live
  state.

## 25. Final status

**COMPLETE.** Every Milestone 6 acceptance criterion has direct implementation,
test, benchmark, migration, or safe smoke evidence. No later-milestone feature
was introduced.
