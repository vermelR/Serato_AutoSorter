# SeratoAI architecture

## Module map

```text
app.py
  └─ serato_ai.ui.application
       ├─ ui.settings_components
       ├─ ui.manual_page / ui.watcher_page
       ├─ ui.predictions_page / ui.result_components
       └─ ui.session_state
            ↓
       services.prediction_service
       services.crate_assignment_service
       services.watcher_service
       services.model_service
            ↓
       core.models, core.dataframes, core.validation,
       core.assignment_utils, core.confidence, core.crate_filters,
       core.path_utils, core.result_summary
            ↓
       infrastructure adapters and stable low-level modules
       (Serato binary writer/reader, tags, models, watcher queue)
```

`app.py` is intentionally only the Streamlit entry point. It imports and calls
`render_application`; it has no binary writer, model, filesystem, or approval
rules.

## Personalized onboarding and model lifecycle

On first launch, `ui.application` checks `OnboardingService` before rendering
the prediction workspace. A valid application-managed model (or a valid legacy
configured model) opens the normal workspace. Without one, the user sees the
local-only onboarding flow instead:

1. Confirm the detected or custom Serato root and one or more music folders.
2. `LibraryScanService` validates the root and reads each crate through the
   existing binary parser. It never writes a crate file.
3. The scan preserves raw `%%` crate names, matches audio paths without
   guessing between duplicate filenames, reports missing/ambiguous files, and
   classifies training eligibility. Workflow/inbox crates are excluded by
   complete crate-name components; DJs can override inclusions/exclusions.
4. `TrainingDatasetService` makes one record per physical track with every
   eligible crate label. Latest successful Final Crates feedback overrides
   imported membership for that track on a future build.
5. `FeatureExtractionService` owns an indexed SQLite feature cache in the
   application data directory. Unchanged files reuse `float32` vectors through
   per-key lookup; modified files, schema changes, corrupt stores, and isolated
   failures recover safely without loading the cache into Streamlit or
   modifying audio/tags. Resource modes bound extraction to 1, 2, or 4 workers.
6. `TrainingService` evaluates a bounded compact search. The selected
   `SharedKNNMultiLabelClassifier` stores one shared normalized `float32`
   feature matrix and a sparse label matrix; the compact linear alternative
   remains available in the search. Candidate evaluation is validation-only,
   and fingerprint/configuration-matched preparations and completed candidate
   evaluations are reused from the external training cache.
7. `ModelActivationService` saves the candidate, reloads it, smoke-tests valid
   probabilities, and atomically replaces only the active pointer after the
   configured quality gate passes. The previous active pointer stays available
   for restore and a failed candidate changes neither pointer.

Application paths are calculated by `infrastructure.application_data` using a
portable per-user application-data location (or `SERATO_AI_DATA_DIR`). Active
and previous manifests point to immutable files under `models/versions`;
unactivated files remain under `models/candidates`; metadata, feature cache,
training cache, scan index, background jobs, feedback, evaluation arrays,
quality/performance history, and benchmark reports are separate
application-owned stores. They never live inside the Serato library.

Personalized bundles use `prediction_semantics = independent_multilabel`.
Their crate probabilities are independently ranked and deliberately do not sum
to 100%. Legacy single-label model bundles keep their previous conditional,
renormalized filtering semantics.

`ModelHealthService` aggregates setup state, active/previous/candidate sizes,
feature/training caches, feedback, quality/evaluation/benchmark storage, load
and prediction diagnostics, scan outcomes, previous-model availability,
reclaimable storage, and model-size warnings. Update Model rescans local
sources first and reuses index/cache entries. Interrupted setup remains
resumable. A normal legacy bundle can be imported without deleting its source;
an oversized legacy artifact is never deserialized for migration.

## Milestone 6 model and storage design

The 33,878,876,869-byte legacy `serato_model.pkl` is a 400-tree unbounded
multiclass `RandomForestClassifier`. Scikit-learn stores a dense class-count
vector at every node (`trees × nodes × classes`). With 919 classes,
`tree_.value` dominates the serialized graph. Disposable reproductions showed
17.84 MiB of 19.28 MiB for 2,000 rows/100 classes/10 trees and 74.49 MiB of
77.49 MiB for 4,000 rows/200 classes/10 trees. No source bundle path embeds raw
audio, feature-cache history, or a training DataFrame; the dense node values
are the exact growth mechanism.

`PersonalModelStore` writes a compressed `seratoai-inference-v1` bundle with:

- estimator and raw `%%` class names;
- feature columns/schema and independent multi-label semantics;
- global/per-crate thresholds and compact quality configuration;
- model/bundle version, estimator type, creation time, compression method;
- exact artifact size, stored both at the bundle top level and in compact
  inference metadata.

Training rows/matrices, feature cache, feedback, full evaluation rows, quality
history, candidate search history, service objects, scan reports, and prior
models are forbidden from the inference bundle. `PersonalModelStore` strips
known training-only keys and writes full `ModelVersionMetadata` externally.
Full prediction-evaluation rows are compressed owner-only `.npz` files; the
bounded quality history contains summaries and references.

`StorageBudgets` defaults to 500 MiB preferred, 2 GiB warning, 5 GiB automatic
activation review block, and 8 GiB hard block. A candidate is serialized and
measured before reload or activation. Only an explicit developer override can
pass the review/hard block. Activation then reloads, smoke-predicts, checks
finite compatible probabilities, applies the unchanged quality gate, moves
the candidate into immutable version storage, and atomically advances
active/previous manifests.

`ModelService` caches a loaded bundle by resolved path, mtime, size, and loader
identity under a lock. Reruns reuse the same object and do not create duplicate
model instances. A version/file change reloads; a corrupt active version falls
back read-only to the prior known-good artifact without changing manifests.
Prediction extraction is separated from inference and estimator calls use
bounded batches (128 by default), preserving rankings, thresholds, confidence,
review reasons, category filters, and manual/watcher results.

## Incremental work and responsiveness

`ScanIndexStore` persists crate and music signatures by library scope.
Unchanged crates reuse their prior parsed records; changed crates parse once;
a malformed changed crate retains its prior valid record and is retried later.
Music indexing records path, size, mtime, device/inode, and directory
signatures. An incremental pass avoids enumerating unchanged directories,
detects content changes, adds/deletes, and inode-preserving moves, and produces
the same fingerprint and matches as a clean full scan. Missing external roots
retain cached entries and are marked unavailable rather than deleted.

`BackgroundJobManager` bounds long work to at most two jobs and persists the
latest 100 snapshots. It reports stage, count, percentage, elapsed time, ETA,
warnings, diagnostics, cancellation, checkpoints, and a resume checkpoint
where supported. Scan, training/evaluation, cache compaction, and oversized
legacy migration use it. Cancellation is checked before activation, so an
incomplete model cannot advance the active pointer.

`PerformanceStore` keeps the latest 500 local-only diagnostics, including
operation, duration, item/cache counts, worker count, memory estimate, model
size, warnings, and failure reason. It receives model-load, batch-prediction,
feature-extraction, full/incremental-scan, and training/evaluation events. No
analytics or network transport exists.

The watcher confirms a supported file is present, readable, non-temporary, and
stable across samples before analysis. Paths and device/inode identities
deduplicate pending/processed events; crate signatures avoid reparsing
unchanged inbox crates; transient drive errors retry with bounded backoff. A
single SQLite transaction handles bursts and caps the queue at 10,000 entries.
Search, view counts, category signatures, pagination, and review edits execute
in SQLite. Streamlit holds only the current page, and Final Crates remain raw
multi-select values in stored payloads.

Manual predictions are paginated/filterable in memory after the explicit
Generate action. Neither manual nor watcher review controls call model load,
scan, extraction, prediction, training, or evaluation. The global immutable
model cache is an additional guard against Streamlit rerun reloads.

## Retention and cleanup

Default retention preserves the active model, one previous known-good model,
legacy reference/source, recent quality history, referenced evaluation files,
feedback, current feature cache, training cache, scan index, and interrupted
job information. Rejected candidates, older immutable model versions,
unreferenced evaluation/report artifacts, and interrupted `.tmp` files are
reported as reclaimable. Cleanup is explicit and confirmation-gated;
protected active/previous/legacy artifacts cannot be selected by those
operations.

## Model-quality evaluation and activation

M5 adds a local, reproducible quality pipeline around the personalized
one-vs-rest model. `TrainingService` gives physical-track groups to
`ModelQualityService`; canonical identities and stable file fingerprints are
grouped before splitting, so a duplicate path, crate reference, feedback row,
or cache record cannot appear across train, validation, and test roles.
Training fits model parameters only on train rows; bounded configuration search
and per-crate F1 threshold tuning use validation rows only; the selected
challenger is evaluated once on the untouched test rows. Small datasets use a
clearly labeled train/validation-only fallback rather than inventing a test
score.

The quality artifact records a hashed dataset fingerprint, split assignments,
random seed, schema/feature versions, estimator parameters, thresholds,
multi-label metrics, ranking metrics, calibration bins, per-crate support and
health, hierarchy errors, ablations, comparison, and gate decision. It contains
hashed track identities only—never raw source-audio paths or audio data.

The deterministic most-frequent baseline is the first-model champion. A later
candidate is compared with the active model on the same final split whenever
its schema is compatible. `QualityGateResult` rejects leakage, invalid schema
or probabilities, coverage loss, unusable suggestions, material micro/macro
F1 or Recall@3 regression, calibration regression, and supported rare-crate
recall collapse. Candidate snapshots remain local diagnostics; only a passing
candidate can reach the existing atomic activation step. The active and
previous models are never overwritten by a rejected candidate.

`ModelQualityStore` keeps the most recent 20 active/rejected snapshots
atomically under the application data reports directory. Summary JSON remains
compact; referenced full prediction records use compressed owner-only `.npz`
arrays. Model Health reads the active, previous, and latest rejected snapshots
and exports a portable summary report.
Older M4 bundles use conservative default thresholds and show that a quality
snapshot is unavailable until an Update Model run; they are never forced to
retrain merely to open the application.

## Offline-first metadata

`MetadataEnrichmentService` is the single read-only metadata contract for
manual prediction, both watcher modes, and personalized-training extraction.
It reads `EmbeddedTagProvider` before audio features or crate inference, so a
valid embedded Genre and Year/Date remain authoritative. `LocalGenreProvider`
is called only when normalized embedded Genre is missing; it returns a
confidence and never predicts Year. Missing or invalid Year remains blank.

`AcoustIDProvider` and `MusicBrainzProvider` are disabled infrastructure
adapters in this release. They expose stable provider/result interfaces and
report a disabled status, but make no network request and require no API key.
The service carries source, confidence, raw Year, warnings, provider status,
and `online_lookup_attempted=False` through prediction and training records.
A later update can register enabled providers in the same service without
redesigning onboarding, watcher queues, manual prediction, or training.

All metadata, feature cache entries, feedback, model metadata, and scan reports
stay local. The metadata strategy version is stored with every personalized
model; Model Health displays the plain-language strategy: existing tags first,
local ML for missing Genre, blank missing Year, and online providers disabled.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| `serato_ai.ui` | Streamlit widgets, layout, sidebar input, result presentation, and session-state access. |
| `serato_ai.services` | Reusable prediction, Apply, model, and watcher lifecycle coordination. Services return typed responses and never render Streamlit messages. |
| `serato_ai.core` | Deterministic domain models, crate filtering, probability handling, dataframe conversion, validation, assignment expansion, paths, and outcome summaries. |
| `serato_ai.infrastructure` | Adapters for safe Serato binary reads/writes, tag writes, model persistence, and configuration. Existing stable writer/parser modules remain the implementation boundary. |
| `serato_ai.settings` | Typed environment-backed application defaults. `config.py` is a compatibility facade for existing integrations. |
| `services.onboarding_service`, `library_scan_service`, `training_dataset_service`, `feature_extraction_service`, `training_service`, `model_evaluation_service`, `model_activation_service`, `model_health_service`, `feedback_service` | Local onboarding, read-only library scan, multi-label dataset/training, safe candidate activation, health, and future-learning feedback. |

The dependency direction is **UI → Services → Core/Infrastructure**. Core
never imports Streamlit or infrastructure. Infrastructure never imports UI.
Services never import `app.py`.

## Typed contracts

`core.models` provides stable dataclasses for crate suggestions, track
predictions, approved tracks, crate assignments, tag and crate write results,
validation issues, operation summaries, prediction requests/responses, and
Apply requests/responses. Dataframe conversion helpers preserve the existing
review/export columns while raw `%%` crate names remain the writing contract.

## Data flow

### Manual prediction

1. The sidebar creates `RuntimeSettings`.
2. An explicit Generate action calls `PredictionService` once.
3. The service loads the configured model, collects files, performs inference,
   applies category/exclusion filtering, and returns typed predictions.
4. UI dataframe helpers produce the existing review table; Final Crates
   selection state is stored only under the manual state keys.
5. For an M5 bundle, the same post-filter quality configuration supplies the
   per-crate threshold, top-probability margin, support count, and structured
   review reason. It never removes manual Final Crates choices.

### Watcher prediction

1. `WatcherService` constructs the background watcher with the same
   `PredictionService` used by manual scans.
2. Stable, supported, readable candidates receive the same allowed/excluded
   crate rules and produce the same raw prediction schema in the bounded
   SQLite queue.
3. The watcher page asks SQLite for only the selected search/filter page,
   persists Approve/Genre/Year/Final Crates edits for that page, and applies
   only the displayed reviewed page. Folder and Serato-inbox candidates use
   the exact same `PredictionService` quality fields as manual scans.

### Apply to Serato

1. Both pages turn approved dataframe rows into `ApprovedTrack` values.
2. `CrateAssignmentService` validates Final Crates and allow-list membership.
3. It tags each physical track once, expands/deduplicates every raw Final
   Crates assignment, then invokes the safe Serato writer.
4. It returns `ApplyResponse` with independent tag/crate results and an
   `OperationSummary`; the UI decides how to display it.

Only Final Crates determines destinations. Top Suggested Crate and Crate
Suggestions are informational display values.

## Configuration and session state

`ApplicationSettings` derives portable defaults from the active user's home
and environment variables. Normal prediction-sidebar entries remain runtime
only; confirmed onboarding folders, eligibility choices, and setup progress are
persisted separately in the application-owned onboarding store. No runtime
Python module contains a user-specific `/Users/diora/` path.

`ui.session_state` owns all key names and lifecycle methods. Manual predictions,
manual failures, manual selections, the current watcher page, watcher
selections, watcher instance, crate options, background job ids, and filter
signatures are distinct. The complete watcher queue is disk-backed, not a
session-state DataFrame. Resetting manual prediction state cannot erase watcher
state, and resetting queue state cannot erase manual selections.

## Testing strategy

Tests use temporary Serato roots and mockable service dependencies. Core tests
prove pure rules without Streamlit; service tests prove deterministic prediction
and Apply contracts; AppTests cover rendering and truthfulness of messages;
writer integration tests prove binary safety. The production-path and socket
guards in `tests/conftest.py` apply to every test.

## Rules for future features

- UI code must not implement low-level Serato file writing.
- Core modules must not import Streamlit or perform filesystem/network I/O.
- Manual and watcher workflows must use the same prediction and Apply services.
- Services return structured responses instead of rendering UI messages.
- Raw crate names and display labels must remain separate.
- New infrastructure integrations must be mockable.
- No user-specific paths may be hardcoded.
- Every production bug requires a regression test; tests never touch a real
  Serato library.
- Setup and training must never write Serato crates, source audio, or tags.
- Metadata enrichment must read embedded tags first; no live metadata provider
  may be enabled without an explicit future product change and tests.
- Packaging, accounts, cloud synchronization, and installers remain separate
  milestones.
