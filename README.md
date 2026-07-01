# SeratoAI (CrateAI)

Automatically sorts new/downloaded music into the right Serato crates, figures
out each track's real genre from how it sounds, and fills in the release
year — all reviewed by you before anything is written.

## What it does

1. **Crate sorting** — a RandomForest model (trained on your existing Serato
   crates, `phase1_harvest.py` → `phase2_analyze.py` → `phase3_train.py`)
   predicts which crate a new track belongs in from its audio features
   (BPM, brightness, energy, MFCCs).
2. **Genre identification** (`identify.py`, `genre_model.py`) — for each
   track, in priority order:
   - Fingerprints the audio (Chromaprint/AcoustID) and looks up the
     *original recording* in MusicBrainz to get its real genre and release
     year — this is genre "based on how it sounds and the original song."
   - If there's no confident fingerprint match (offline, no API key, or the
     track just isn't in MusicBrainz), falls back to a local ML genre
     classifier trained on your own tagged library.
   - If neither is available, falls back to whatever genre/year tag the
     file already has.
3. **Year tagging** (`tag_writer.py`) — writes the identified genre/year
   directly into the audio file's tags (ID3/Vorbis/MP4, via mutagen).
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

### Genre fingerprinting (optional but recommended)

1. Get a free API key at https://acoustid.org/api-key
2. Install `fpcalc` (Chromaprint) — `brew install chromaprint` on macOS, or
   place a static binary at `vendor/fpcalc` to bundle it like `vendor/ffmpeg`.
3. Set it via the "AcoustID API key" field in the app sidebar, or:
   ```bash
   export ACOUSTID_API_KEY=your_key_here
   ```

Without a key, genre/year identification still works — it just skips
fingerprinting and uses the local ML classifier / existing tags instead.

### Local genre classifier (fallback, optional)

Train it once from a folder of music that already has genre tags:

```bash
python genre_model.py /path/to/tagged/music --recursive
```

This writes `serato_genre_model.pkl`, used automatically as a fallback.

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
  found; review and Apply the same way.
- Turn off **Dry run** in the sidebar once you're happy with results, so
  writes actually happen (a Serato backup is made first by default).

## Training/updating the crate model

```bash
python phase1_harvest.py     # scan existing Serato crates -> serato_training_data.csv
python phase2_analyze.py     # extract audio features -> music_features_dataset.csv
python phase3_train.py       # train -> serato_model.pkl
```
