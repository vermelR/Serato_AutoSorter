from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any
from urllib.parse import unquote

import joblib
import librosa
import numpy as np
import pandas as pd
from mutagen import File as MutagenFile

import config
import genre_model
import identify

# -------------------------
# Configuration
# -------------------------
AUDIO_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a"}
WINDOW_SECONDS = 30
OFFSET_SECONDS = 60

# Reduce noisy librosa warnings in terminal
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# -------------------------
# Path helpers
# -------------------------
def normalize_path(raw_path: str) -> str:
    if raw_path is None:
        return ""

    p = str(raw_path).strip().replace("\r", "").replace("\n", "")
    p = p.strip().strip('"').strip("'")

    if p.startswith("file://"):
        p = p.replace("file://", "", 1)

    p = unquote(p)
    p = os.path.expanduser(p)

    if p.startswith("Users/"):
        p = "/" + p

    # Handle legacy "Mac:Users:..." style paths
    if ":" in p and not p.startswith("/"):
        parts = p.split(":")
        if "Users" in parts:
            idx = parts.index("Users")
            p = "/" + "/".join(parts[idx:])

    return p


def collect_audio_files(inputs: list[str], recursive: bool = True) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        p = Path(normalize_path(item))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()

        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            found.append(p)
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in it:
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    found.append(f)

    # de-dupe
    seen = set()
    uniq = []
    for f in found:
        s = str(f)
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq


# -------------------------
# ffmpeg decoding (bundled)
# -------------------------
def bundled_ffmpeg_path() -> Path:
    """
    Expects ffmpeg at: project_root/vendor/ffmpeg
    When packaged, we also bundle it to vendor/ffmpeg.
    """
    return (Path(__file__).parent / "vendor" / "ffmpeg").resolve()


def decode_to_wav(input_path: str) -> str:
    """
    Decode any audio format to a temporary WAV using bundled ffmpeg.
    This makes decoding consistent on any laptop (no installs).
    """
    ffmpeg = bundled_ffmpeg_path()
    if not ffmpeg.exists():
        raise FileNotFoundError(
            f"Bundled ffmpeg not found at {ffmpeg}. "
            "Put ffmpeg at SeratoAI/vendor/ffmpeg"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        str(ffmpeg),
        "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "44100",      # sample rate
        tmp.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp.name


# -------------------------
# Feature extraction
# -------------------------
def extract_audio_features(file_path: str) -> list[float]:
    """
    Features must match training order:
    [bpm, brightness, energy, mfcc_1..mfcc_13]
    """
    wav_path = None
    try:
        wav_path = decode_to_wav(file_path)

        # Prefer the "meat" at 60s, else fallback to start
        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True, offset=OFFSET_SECONDS, duration=WINDOW_SECONDS)
        except Exception:
            y, sr = librosa.load(wav_path, sr=None, mono=True, offset=0, duration=WINDOW_SECONDS)

        # 1) BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # 2) Brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(centroid))

        # 3) Energy
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # 4) MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0).astype(float).tolist()

        return [float(tempo), brightness, energy] + mfccs_scaled

    finally:
        # Cleanup temp wav to avoid filling disk
        if wav_path:
            try:
                os.remove(wav_path)
            except Exception:
                pass


def load_model_bundle(model_path: str = "serato_model.pkl") -> dict:
    p = Path(normalize_path(model_path))
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    bundle = joblib.load(str(p))
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model bundle missing keys: 'model' and/or 'feature_columns'")
    return bundle


# -------------------------
# Metadata extraction (clean table)
# -------------------------
def _first_tag(tags: dict, keys: list[str]) -> str:
    for k in keys:
        if k in tags:
            val = tags.get(k)
            if isinstance(val, (list, tuple)) and val:
                return str(val[0]).strip()
            if val is not None:
                return str(val).strip()
    return ""


def parse_artist_title_from_filename(stem: str) -> Tuple[str, str]:
    s = stem.replace("_", " ").strip()

    if " - " in s:
        artist, title = s.split(" - ", 1)
        return artist.strip(), title.strip()

    return "", s


def read_track_metadata(path: Path) -> Dict[str, Any]:
    artist = ""
    title = ""
    genre = ""
    year = ""

    try:
        audio = MutagenFile(str(path), easy=True)
        if audio is not None and getattr(audio, "tags", None):
            tags = audio.tags
            artist = _first_tag(tags, ["artist", "albumartist"])
            title = _first_tag(tags, ["title"])
            genre = _first_tag(tags, ["genre"])
            year = _first_tag(tags, ["date", "year"])

            if year and len(year) >= 4:
                year = year[:4]
    except Exception:
        pass

    if not title or not artist:
        fa, ft = parse_artist_title_from_filename(path.stem)
        if not artist:
            artist = fa
        if not title:
            title = ft

    return {
        "artist": artist.strip(),
        "title": title.strip(),
        "genre": genre.strip(),
        "year": year.strip(),
    }


# -------------------------
# Genre / Year identification
# -------------------------
def identify_genre_year(path: Path, feats: list[float], existing_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Figure out the best genre/year for a track, in priority order:
      1. Audio fingerprint match (AcoustID -> MusicBrainz) - identifies the
         *original* recording from how it sounds, and reads its real genre/year.
      2. Local ML genre classifier (trained on your own tagged library) - used
         when there's no confident fingerprint match. Only supplies genre,
         not year.
      3. Whatever genre/year tag was already on the file.

    Returns genre/year plus which source each one came from, so the UI can
    show how confident to be in the suggestion.
    """
    genre = ""
    year = ""
    genre_source = "none"
    year_source = "none"

    fp_result = None
    try:
        fp_result = identify.lookup_by_fingerprint(str(path))
    except Exception:
        fp_result = None

    if fp_result is not None:
        if fp_result.genre:
            genre, genre_source = fp_result.genre, "fingerprint"
        if fp_result.year:
            year, year_source = fp_result.year, "fingerprint"

    if not genre:
        try:
            ml_genre, ml_conf = genre_model.predict_genre(feats)
        except Exception:
            ml_genre, ml_conf = None, 0.0
        if ml_genre and ml_conf >= config.GENRE_MODEL_MIN_CONFIDENCE:
            genre, genre_source = ml_genre, f"ml_model ({ml_conf:.0%})"

    if not genre and existing_meta.get("genre"):
        genre, genre_source = existing_meta["genre"], "tag"

    if not year and existing_meta.get("year"):
        year, year_source = existing_meta["year"], "tag"

    return {
        "genre": genre,
        "year": year,
        "genre_source": genre_source,
        "year_source": year_source,
    }


# -------------------------
# Prediction proposals (clean)
# -------------------------
def propose_crates_for_files(
    bundle: dict,
    files: list[Path],
    topk: int = 3,
    identify_genre: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    rows = []
    fails = []

    for f in files:
        try:
            meta = read_track_metadata(f)
            feats = extract_audio_features(str(f))

            X = pd.DataFrame([feats], columns=feature_cols)

            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            class_probs = sorted(zip(model.classes_, probs), key=lambda t: t[1], reverse=True)

            conf_map = dict(class_probs)
            conf = float(conf_map.get(pred, 0.0))

            if identify_genre:
                gy = identify_genre_year(f, feats, meta)
            else:
                gy = {
                    "genre": meta["genre"], "year": meta["year"],
                    "genre_source": "tag", "year_source": "tag",
                }

            row = {
                "Song Title": meta["title"] or f.stem,
                "Artist": meta["artist"],
                "BPM": round(float(feats[0]), 2),
                "Genre": gy["genre"],
                "Genre Source": gy["genre_source"],
                "Year": gy["year"],
                "Year Source": gy["year_source"],
                "Suggested Crate": pred,
                "Confidence": round(conf, 2),
                "path": str(f),  # keep hidden for commit
            }

            for i, (label, pval) in enumerate(class_probs[:topk], start=1):
                row[f"_top{i}_crate"] = label
                row[f"_top{i}_prob"] = float(pval)

            rows.append(row)

        except Exception as e:
            fails.append({
                "Song Title": f.stem,
                "path": str(f),
                "error": f"{type(e).__name__}: {e}",
            })

    return pd.DataFrame(rows), pd.DataFrame(fails)
