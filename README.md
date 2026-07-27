# SeratoAI (CrateAI)

Automatically sorts new/downloaded music into the right Serato crates and
prepares their existing Genre and Year metadata for review — all locally and
reviewed by you before anything is written.

## What it does

1. **Crate sorting** — a compact personalized multi-label model learns from
   your existing Serato crates and ranks independent crate probabilities from
   scalar audio features (BPM, brightness, energy, MFCCs). A track can belong
   to several crates, and you can always replace the suggestions with one or
   more raw **Final Crates** before approval.
2. **Offline-first Genre and Year metadata** — before audio analysis or model
   prediction, SeratoAI reads the file's existing tags. A valid embedded Genre
   and a valid embedded Year/Date are preserved. Only a missing or invalid
   Genre may use the local genre model, and only when its confidence clears the
   configured threshold. A missing or invalid Year stays blank. This release
   makes no AcoustID, MusicBrainz, fingerprint, or other online metadata
   request.
3. **Optional tag writing** (`tag_writer.py`) — after you approve a track,
   Apply can write the reviewed genre/year directly into the audio file's tags
   (ID3/Vorbis/MP4, via mutagen). Onboarding and training never modify tags.
   Serato reads Genre/Year straight from the file's tags, so this is what
   makes them show up correctly in Serato.
4. **Background watcher** (`watcher.py`) — polls your configured "new
   music" folders (e.g. Downloads) and/or a Serato inbox crate for new
   tracks, and automatically runs the whole prediction pipeline on them in
   the background. Results queue up in the app for you to review — nothing
   is written to Serato or your files until you hit **Apply**.

## Setup

```bash
pip install -r requirements.txt
```

Audio decoding uses a bundled `ffmpeg` at `vendor/ffmpeg` (already present).

### Local genre classifier (fallback, optional)

Train it once from a folder of music that already has genre tags:

```bash
python genre_model.py /path/to/tagged/music --recursive
```

This writes `serato_genre_model.pkl`. It is used only when a track has no
valid embedded Genre; it never predicts Year or replaces a valid Genre tag.

The sidebar setting **Identify genre and year automatically** is enabled by
default. It means existing tags first, local Genre fallback only, blank missing
Year, and no online metadata lookup. No API key is needed.

## Running the app

```bash
python run_app.py
```

or in dev mode: `streamlit run app.py`

In the app:

- **Manual scan** — point at a folder, click "Generate Predictions", review
  crate/genre/year suggestions (all fields are editable), approve, Apply.
- **Background watcher** — configure folders/inbox crates in the sidebar and
  click "Start watching". New tracks show up under "Auto-Detected" as they're
  found; review and Apply the same way. The disk-backed queue supports search,
  pending/low-confidence/approved views, page sizes, and page-by-page review
  without loading the complete queue.
- Turn off **Dry run** in the sidebar once you're happy with results, so
  writes actually happen (a Serato backup is made first by default).

## First-time personalized setup

If SeratoAI does not have a working model yet, it opens a guided local setup
instead of asking for a model filename.

1. Confirm the Serato folder (usually `Music/_Serato_`) or choose a custom
   location. Setup only reads crate files; it never changes them.
2. Enter every music folder Serato uses, one per line. Internal disks,
   removable drives, paths with spaces, and Unicode paths are supported.
3. Review the crate scan. Existing crates become training labels; inbox,
   unsorted, review, temporary, and backup workflow crates are excluded by
   default. You can include or exclude crates before training. Crates with too
   few matched tracks remain available for Final Crates selection but are not
   prediction classes.
4. Build the model. Features and the personalized model stay on your computer.
   Scanning, feature extraction, training, evaluation, migration, and cache
   compaction use bounded background work with progress and safe cancellation.
   Interrupted work retains checkpoints and completed cache entries.
5. Review **Model Health** later to see quality, missing files, pending Final
   Crates corrections, cache status, and whether an update is available.

Training uses all eligible crate memberships for a track, so a song can learn
that it belongs in more than one crate. These independent probabilities are
ranked as suggestions and do not need to total 100%. A candidate model must
save, reload, predict, and pass its quality gate before it becomes active; a
failed retraining attempt leaves the last working model in place. Approved live
Final Crates corrections are stored locally and take precedence in the next
training build. Dry runs and failed crate writes are never recorded as learned
feedback.

Existing users can continue using a configured model. A normal-sized legacy
bundle can be imported without deleting its source. An oversized legacy model
is never loaded merely to migrate it: Model Health builds a compact candidate
from the saved scalar feature CSV, runs the unchanged quality/storage/reload
gates, activates it atomically only on success, and retains the original as a
recoverable legacy reference.

No account, cloud upload, analytics, API key, or network metadata service is
required for onboarding or model training. Model Health also states the current
metadata strategy and keeps its version for later migrations.

## Understanding model quality

Model Health records a reproducible local quality snapshot whenever SeratoAI
trains a personalized candidate. It reports Micro and Macro F1, Top-1/Top-3/
Top-5 ranking quality, Precision@K and Recall@K, probability calibration,
low-confidence review rate, crate-by-crate health, and the thresholds used for
each crate. Top-3 is especially useful for DJs: the correct crate can be a
valuable reviewed suggestion even when it is not the first result.

Crate probabilities are independent multi-label estimates, so they do not add
to 100%. A low-confidence or close-call result remains available for you to
review and choose Final Crates manually; it is never automatically written.
Some crates need more examples when there is not enough held-out support to make
a trustworthy claim.

Candidates are chosen using validation data and compared with the active model
or a deterministic baseline on a separate test split where the library is large
enough. A candidate may be rejected for leakage, weaker ranking or recall, poor
calibration, rare-crate regression, or invalid output—even with higher training
accuracy. The prior model remains active, and successful Final Crates feedback
contributes to later retraining without overweighting duplicate corrections.

## Storage, performance, and cleanup

The active `seratoai-inference-v1` bundle contains only prediction-time state:
the estimator, raw crate names, feature schema, thresholds, quality
configuration, compact identity metadata, and exact artifact size. Training
matrices, feature cache, feedback, full evaluation records, quality history,
candidate-search cache, rejected models, and previous versions are separate
local stores. Full evaluation rows are compressed `.npz`; feature and watcher
stores use SQLite incremental lookup.

The compact estimator stores one shared `float32` feature matrix and one sparse
multi-label matrix instead of a class-width value vector at every tree node.
Model files are compressed, versioned, measured before activation, and loaded
once per immutable version. Predictions are extracted first and inferred in
bounded batches.

Default model budgets are:

- Preferred: at most 500 MiB
- Warning: above 2 GiB
- Automatic-activation review block: above 5 GiB
- Hard block: above 8 GiB

Only a deliberate developer override can bypass the review/hard block. Use
`SERATO_PREFERRED_MODEL_BYTES`, `SERATO_MODEL_WARNING_BYTES`,
`SERATO_MODEL_REVIEW_BYTES`, and `SERATO_MODEL_HARD_LIMIT_BYTES` to configure
budgets. `SERATO_ALLOW_OVERSIZED_MODEL=true` is intended for controlled
development only.

Model Health reports active/previous/candidate sizes, feature and training
caches, quality/evaluation/feedback/benchmark storage, load and prediction
latency, cache hit rate, the last incremental scan, warnings, and reclaimable
space. Cleanup requires an explicit confirmation and protects the active model,
one previous known-good model, legacy source, feedback, recent quality history,
and live feature cache.

`SERATO_PERFORMANCE_MODE` accepts `low_resource`, `balanced`, or `fast` for
bounded feature extraction (1, 2, or at most 4 workers).
`SERATO_PREDICTION_BATCH_SIZE` controls inference batches and
`SERATO_TABLE_PAGE_SIZE` controls the initial review page.

## External and removable drives

Missing or sleeping music folders are reported clearly. Their prior scan index
and cache entries remain available; SeratoAI does not infer deletions or purge
features merely because a drive is disconnected. Reconnect the same path and
run the incremental scan to resume. Scans and extraction run in cancellable
background jobs, and transient watcher failures use bounded retry backoff.

## Benchmarks

The repeatable local benchmark covers 500/20, 10,000/100, and 50,000/300
track/crate profiles. It measures startup, model load, single/batch prediction,
full/incremental scans, cache reads/writes, watcher bursts, training,
evaluation, table preparation, memory, and artifact size. It uses generated
scalar features, temporary empty audio files, temporary Serato roots, and no
network:

```bash
.venv/bin/python -m benchmarks.m6_benchmarks \
  --profiles small,medium,large \
  --output benchmarks/m6-results.json
```

See [MILESTONE6_REPORT.md](MILESTONE6_REPORT.md) for the root-cause evidence,
compact-model comparison, quality result, storage breakdown, and recorded
benchmark results.

## Legacy training scripts

```bash
python phase1_harvest.py     # scan existing Serato crates -> serato_training_data.csv
python phase2_analyze.py     # extract audio features -> music_features_dataset.csv
python phase3_train.py       # train -> serato_model.pkl
```

These scripts remain for compatibility and reproducibility. New personalized
training should use the guided app flow, which applies the compact model,
frozen-split quality gate, versioned storage, rollback, and size gates.

## Development rule

Every fixed production bug must receive a regression test before the fix is
considered complete. See [TESTING.md](TESTING.md) for the temporary-only test
fixtures, safety guard, marker suites, coverage command, and failure diagnosis.
