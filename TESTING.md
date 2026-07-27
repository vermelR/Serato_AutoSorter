# Testing SeratoAI

SeratoAI's test suite is deliberately isolated from any installed Serato
library. It never reads or writes `/Users/diora/Music/_Serato_`, and it also
blocks the path configured in `SERATO_PRODUCTION_ROOT` if one is supplied.
The autouse guard in `tests/conftest.py` intercepts common `Path`, `os`,
`shutil`, and built-in file operations before they can reach either location.
It also disables outbound socket connections. Milestone 4 metadata tests prove
that the disabled AcoustID and MusicBrainz adapters never make a live request.

## Install

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Use the project environment when it exists (`.venv/bin/python` on macOS/Linux);
the commands below use `python` for portability.

## Run tests

Fast isolated tests (skips integration and slow tests):

```bash
python -m pytest -q -m "not slow and not integration"
```

Full suite:

```bash
python -m pytest -q
```

Writer/parser suite:

```bash
python -m pytest -q -m writer
```

Streamlit AppTest suite:

```bash
python -m pytest -q -m streamlit
```

Milestone 4–6 focused groups:

```bash
python -m pytest -q -m onboarding
python -m pytest -q -m training
python -m pytest -q -m cache
python -m pytest -q -m model_store
python -m pytest -q -m model_health
python -m pytest -q -m metadata
python -m pytest -q -m model_quality
python -m pytest -q -m metrics
python -m pytest -q -m leakage
python -m pytest -q -m calibration
python -m pytest -q -m thresholds
python -m pytest -q -m candidate_comparison
python -m pytest -q -m performance
python -m pytest -q -m model_storage
python -m pytest -q -m model_size
python -m pytest -q -m incremental_scan
python -m pytest -q -m background_jobs
python -m pytest -q -m batch_prediction
python -m pytest -q -m migration
```

Run coverage (terminal report plus `htmlcov/index.html`):

```bash
python -m pytest -q --cov --cov-report=term-missing --cov-report=html --cov-fail-under=65
```

The 65% branch-coverage floor is a minimum, not the Milestone 6 target. Compare
the final report with the recorded pre-change 82.97% total and investigate a
meaningful regression. The writer and path-safety modules should remain
especially high; Streamlit rendering is covered by focused AppTests and pure
helper tests rather than fragile browser CSS checks.

## Milestone 6 validation

Compile the public entry points required by the specification:

```bash
python -m py_compile app.py phase4_engine.py serato_writer.py watcher.py tag_writer.py
python -m compileall -q serato_ai benchmarks
```

Run the exact regression matrix:

```bash
python -m pytest -q
python -m pytest -q -m writer
python -m pytest -q -m streamlit
python -m pytest -q -m "not slow and not integration"
python -m pytest -q tests/test_milestone6_performance.py
python -m pytest -q --cov --cov-report=term-missing --cov-report=html --cov-fail-under=65
```

Run import smoke checks:

```bash
python -c "import app, phase4_engine, watcher, serato_writer, tag_writer"
python -c "from serato_ai.services.compact_estimators import SharedKNNMultiLabelClassifier"
```

Run all repeatable synthetic benchmark profiles:

```bash
python -m benchmarks.m6_benchmarks \
  --profiles small,medium,large \
  --output benchmarks/m6-results.json
```

The benchmark must use only generated scalar features, temporary empty audio
fixtures, temporary Serato roots, and local stores. It records startup,
cold/warm model load, single/batch inference, full/incremental crate and music
scans, cache access, watcher bursts, training preparation/candidate fit,
evaluation, table preparation, memory, and artifact sizes. Avoid exact timing
assertions; verify work counts, bounds, warm-load disk-call count, one batch
estimator call, zero unchanged directory enumeration, queue capacity, page
size, and incremental/full fingerprint equality.

`benchmarks/m6_compact_model_comparison.py` and
`benchmarks/m6_bounded_tree_comparison.py` are exploratory frozen-split
comparisons. Their experimental estimators stay in `benchmarks/`, never in the
production service package. Production candidate selection uses validation
only and evaluates the selected model once on the final test split.

## Fixtures and safety

`tests/conftest.py` supplies a disposable Serato root, `SubCrates` directory,
valid and malformed binary crates, copied audio placeholders, a temporary music
library, model file, feature cache, backups, isolated app configuration,
deterministic prediction model/probabilities, watcher queue, and mock writers.
Every test gets a new temporary queue and processed-index path.

The same autouse fixture also supplies a temporary home directory and
`SERATO_AI_DATA_DIR`. Onboarding detection, application-managed models,
feature-cache files, feedback, and reports must therefore be tested only under
`tmp_path`; tests must never probe an installed Serato library or application
data folder. M4 service tests use tiny disposable audio bytes and mock feature
extractors, not real music analysis.

Tests must use those fixtures or `tmp_path`; never reference a real Serato root,
personal music folder, API key, or network service. Metadata fixtures must prove
that embedded Genre/Year wins, local ML runs only for missing Genre, missing or
invalid Year remains blank, provider statuses remain disabled, and source audio
bytes are unchanged. The production-path test is intentional: it proves the
guard fails before access occurs.

M5 quality fixtures use deterministic synthetic multi-label matrices and
temporary model/history stores. They must assert split ownership before model
fit, keep threshold and candidate selection on validation data only, keep the
final test split untouched, and label limited-data validation-only results
honestly. Quality-history fixtures contain hashed identities only; they must not
write source audio, crate bytes, or local music paths into exported artifacts.

M6 fixtures must also assert:

- exact serialized size metadata and pre-activation budget gates;
- no training matrices/cache/history keys in inference bundles;
- active/previous/candidate/legacy physical separation and cleanup protection;
- model load once per immutable version and prior-version fallback on
  corruption;
- batch versus individual ranking/confidence/threshold/review equivalence;
- SQLite feature-cache hit/miss/invalidation/deletion/corruption behavior;
- crate/music add, modify, move, delete, disconnect, reconnect, cancellation,
  and incremental/full equality;
- watcher file stability, path/inode deduplication, bounded transactional
  bursts, SQLite search/filter/page behavior, and processed-track retention;
- background progress, ETA, failure, cancellation, checkpoint, resume, bounded
  history, and no incomplete activation;
- sequential/parallel equality, four-worker cap, failure isolation,
  cancellation, and preservation of completed cache entries;
- Streamlit Final Crates, approval state, search, filters, and pagination
  without model reload, scan, extraction, prediction, evaluation, or training.

Evaluation-record fixtures use owner-only compressed `.npz` files and verify
summary history remains bounded. Candidate-cache fixtures must change either
the dataset or configuration fingerprint and prove stale evaluations are not
reused.

## Architecture tests

The `serato_ai/` package separates pure `core/` rules, reusable `services/`,
filesystem/external `infrastructure/` adapters, and Streamlit `ui/` modules.
Architecture tests verify imports, typed service contracts, session-state
isolation, portable settings, thin `app.py`, and absence of circular imports.

Run just those checks with:

```bash
python -m pytest -q tests/test_architecture_boundaries.py tests/test_core_architecture.py
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency direction and rules
for adding future features.

## Adding a regression test

1. Reproduce the bug using only a fixture or `tmp_path`.
2. Add a focused test to the nearest behavior area and apply the appropriate
   marker (`unit`, `writer`, `prediction`, `watcher`, `streamlit`, `onboarding`,
   `training`, `cache`, `model_store`, `model_health`, `metadata`,
   `integration`, `model_quality`, `metrics`, `leakage`, `calibration`,
   `thresholds`, `candidate_comparison`, `performance`, `model_storage`,
   `model_size`, `incremental_scan`, `background_jobs`, `batch_prediction`, or
   `migration`).
3. Assert both the reported outcome and the no-corruption invariant (bytes,
   queue state, tag mock calls, or raw crate names as applicable).
4. Run the focused marker suite and then the full/coverage commands above.

Every fixed production bug must receive a regression test before the fix is
considered complete.

## Diagnosing failures

- A production-path or network-access assertion means the test accidentally
  escaped its fixtures; replace the path/service with a temporary fixture/mock.
- Parser/writer failures should include a byte range or the preserved source
  error. Verify the original temporary crate bytes did not change.
- Streamlit failures can be run alone with `-m streamlit`; inspect the failed
  widget label/session state rather than relying on browser CSS.
- A model-size failure should report the measured candidate bytes and preserve
  active/previous manifests. Never load the oversized legacy pickle to inspect
  it; use source analysis, file headers, and disposable scaled reproductions.
- An incremental-scan mismatch must be compared with a clean index/full scan
  on the same disposable fixture. A disconnected root must retain cached state,
  not be interpreted as mass deletion.
- A watcher paging failure should inspect SQLite counts and only the requested
  page; loading the entire queue to make the test pass is a regression.
- Open `htmlcov/index.html` after a coverage run to find untested branches in
  core modules.
