import os
import sys
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import librosa
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = "serato_training_data.csv"
OUTPUT_FEATURES = "music_features_dataset.csv"
ERROR_LOG_CSV = "phase2_missing_or_failed.csv"

# Audio window config
WINDOW_SECONDS = 30
OFFSET_SECONDS = 60

# Optional: map old prefixes to new ones if drive names changed
PATH_PREFIX_MAP = {
    # "/Volumes/OldDrive/Music/": "/Volumes/NewDrive/Music/",
}


def normalize_path(raw_path: str) -> str:
    """
    Normalize file paths to avoid common CSV/path issues:
    - trims whitespace/CRLF
    - strips quotes
    - handles file://
    - decodes %20 etc
    - expands ~
    - fixes missing leading slash (Users/... -> /Users/...)
    - fixes classic mac colon paths (Macintosh HD:Users:... -> /Users/...)
    - applies prefix remaps
    """
    if raw_path is None:
        return ""

    p = str(raw_path)

    # Strip whitespace and hidden characters
    p = p.strip().replace("\r", "").replace("\n", "")

    # Strip surrounding quotes
    p = p.strip().strip('"').strip("'")

    # file:// URL -> path
    if p.startswith("file://"):
        p = p.replace("file://", "", 1)

    # Decode URL encoding (spaces etc.)
    p = unquote(p)

    # Expand ~
    p = os.path.expanduser(p)

    # Fix missing leading slash on mac absolute user paths
    # "Users/rohan/..." -> "/Users/rohan/..."
    if p.startswith("Users/"):
        p = "/" + p

    # Fix classic mac colon-separated paths (rare but happens)
    # "Macintosh HD:Users:rohan:Downloads:Music:Track.mp3" -> "/Users/rohan/Downloads/Music/Track.mp3"
    if ":" in p and not p.startswith("/"):
        parts = p.split(":")
        # Try to detect a Users path inside
        if "Users" in parts:
            idx = parts.index("Users")
            p = "/" + "/".join(parts[idx:])

    # Apply prefix remapping (if configured)
    for old_prefix, new_prefix in PATH_PREFIX_MAP.items():
        if p.startswith(old_prefix):
            p = new_prefix + p[len(old_prefix):]
            break

    return p


def pick_audio_window(y: np.ndarray, sr: int) -> np.ndarray:
    """Pick 30s starting at 60s if possible; else first 30s; else whole track."""
    if sr is None or sr <= 0 or y is None or len(y) == 0:
        return y

    start = int(sr * OFFSET_SECONDS)
    end = start + int(sr * WINDOW_SECONDS)
    fallback_end = int(sr * WINDOW_SECONDS)

    if len(y) >= end:
        return y[start:end]
    elif len(y) >= fallback_end:
        return y[:fallback_end]
    else:
        return y


def extract_audio_features(file_path: str, title: str):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)  # full track
        y = pick_audio_window(y, sr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo).reshape(-1)[0])

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(centroid))

        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0).astype(float).tolist()

        return [(tempo), brightness, energy] + mfccs_scaled

    except Exception as e:
        return e


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run phase1_harvest.py first (or cd into the folder).")
        print(f"   CWD: {os.getcwd()}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    if "title" not in df.columns or "path" not in df.columns:
        print("❌ Error: INPUT_CSV must contain columns: title, path (crate optional).")
        print("   Found:", list(df.columns))
        sys.exit(1)

    if "crate" not in df.columns:
        df["crate"] = "Unknown"

    total_tracks = len(df)
    print(f"🎸 Starting analysis on {total_tracks} tracks...")
    print("--------------------------------------------------")

    feature_rows = []
    error_rows = []

    for i, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        crate = str(row.get("crate", "Unknown")).strip()

        raw_path = row.get("path", "")
        norm_path = normalize_path(raw_path)

        # Turn into absolute path if still relative (relative to CSV folder)
        # This is helpful if your CSV has relative paths like "Music/Track.mp3"
        if norm_path and not norm_path.startswith("/"):
            norm_path = str((Path.cwd() / norm_path).resolve())

        p = Path(norm_path)

        print(f"🔄 [{i + 1}/{total_tracks}] Analyzing: {title}...")

        if not norm_path:
            error_rows.append({
                "index": i, "title": title, "crate": crate,
                "raw_path_repr": repr(raw_path),
                "normalized_path": norm_path,
                "reason": "empty_path", "error": ""
            })
            print(f"   ❌ Skipped (empty path). raw={repr(raw_path)}")
            continue

        if not p.exists():
            error_rows.append({
                "index": i, "title": title, "crate": crate,
                "raw_path_repr": repr(raw_path),
                "normalized_path": str(p),
                "reason": "file_not_found", "error": ""
            })
            print(f"   ❌ Missing file. raw={repr(raw_path)} -> normalized={str(p)}")
            continue

        if not p.is_file():
            error_rows.append({
                "index": i, "title": title, "crate": crate,
                "raw_path_repr": repr(raw_path),
                "normalized_path": str(p),
                "reason": "not_a_file", "error": ""
            })
            print(f"   ❌ Not a file: {str(p)}")
            continue

        result = extract_audio_features(str(p), title)

        if isinstance(result, Exception):
            error_rows.append({
                "index": i, "title": title, "crate": crate,
                "raw_path_repr": repr(raw_path),
                "normalized_path": str(p),
                "reason": "feature_extraction_failed",
                "error": f"{type(result).__name__}: {result}"
            })
            print(f"   ⚠️ Feature extraction failed: {type(result).__name__}: {result}")
            continue

        feature_rows.append([title, str(p), crate] + result)

    columns = (
        ["title", "path", "crate", "bpm", "brightness", "energy"]
        + [f"mfcc_{j}" for j in range(1, 14)]
    )

    pd.DataFrame(feature_rows, columns=columns).to_csv(OUTPUT_FEATURES, index=False)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(ERROR_LOG_CSV, index=False)

    print("--------------------------------------------------")
    print(f"✨ DONE! Features for {len(feature_rows)} tracks saved to {OUTPUT_FEATURES}")

    if error_rows:
        print(f"⚠️ {len(error_rows)} tracks were skipped/failed. See {ERROR_LOG_CSV}")
        reasons = pd.Series([r["reason"] for r in error_rows]).value_counts()
        print("Top reasons:")
        for reason, count in reasons.head(5).items():
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
