import os
import sys
import csv
import argparse
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import librosa
import joblib

# -------------------------
# CONFIG
# -------------------------
AUDIO_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a"}
WINDOW_SECONDS = 30
OFFSET_SECONDS = 60


# Optional: if you ever need to map old prefixes to new ones (drive rename etc.)
PATH_PREFIX_MAP = {
    # "/Volumes/OldDrive/Music/": "/Volumes/NewDrive/Music/",
}


# -------------------------
# PATH NORMALIZATION
# -------------------------
def normalize_path(raw_path: str) -> str:
    """
    Normalize paths to avoid common CSV/path issues:
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
    if p.startswith("Users/"):
        p = "/" + p

    # Fix classic mac colon-separated paths
    if ":" in p and not p.startswith("/"):
        parts = p.split(":")
        if "Users" in parts:
            idx = parts.index("Users")
            p = "/" + "/".join(parts[idx:])

    # Apply prefix remapping (if configured)
    for old_prefix, new_prefix in PATH_PREFIX_MAP.items():
        if p.startswith(old_prefix):
            p = new_prefix + p[len(old_prefix):]
            break

    return p


def resolve_to_existing_file(path_str: str) -> Path | None:
    """
    Normalize + resolve + verify existence.
    If still relative, resolve relative to current working directory.
    """
    norm = normalize_path(path_str)
    if not norm:
        return None

    # If relative, resolve from cwd
    if not norm.startswith("/"):
        norm = str((Path.cwd() / norm).resolve())

    p = Path(norm)
    if p.exists() and p.is_file():
        return p
    return None


# -------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------
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


def extract_audio_features(file_path: str):
    """
    Returns features in the same order Phase 2 produced:
    [bpm, brightness, energy, mfcc_1..mfcc_13]
    """
    # Load full track at native sample rate; slice window after load
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = pick_audio_window(y, sr)

    # BPM (tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Brightness (spectral centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = float(np.mean(centroid))

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    # MFCCs (13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0).astype(float).tolist()

    # Keep same ordering as training
    return [float(tempo), brightness, energy] + mfccs_scaled


# -------------------------
# INPUT DISCOVERY
# -------------------------
def collect_audio_files(inputs: list[str], recursive: bool) -> list[Path]:
    """
    Inputs can be:
    - file paths
    - folders (scan for audio)
    """
    found: list[Path] = []

    for item in inputs:
        p = Path(normalize_path(item))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()

        if p.is_file():
            if p.suffix.lower() in AUDIO_EXTS:
                found.append(p)
            continue

        if p.is_dir():
            if recursive:
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                        found.append(f)
            else:
                for f in p.glob("*"):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                        found.append(f)

    # De-dupe while preserving order
    seen = set()
    uniq = []
    for f in found:
        if str(f) not in seen:
            uniq.append(f)
            seen.add(str(f))
    return uniq


def title_from_path(p: Path) -> str:
    return p.stem


# -------------------------
# PREDICT
# -------------------------
def predict_for_files(
    model_bundle: dict,
    files: list[Path],
    topk: int,
    write_probs: bool,
):
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]

    rows = []
    failures = []

    for idx, f in enumerate(files, start=1):
        title = title_from_path(f)
        print(f"🔮 [{idx}/{len(files)}] Predicting: {title}")

        try:
            feats = extract_audio_features(str(f))

            # Build row in the correct order expected by model
            x = pd.DataFrame([feats], columns=feature_cols)
            pred = model.predict(x)[0]

            out = {
                "title": title,
                "path": str(f),
                "predicted_crate": pred,
            }

            if topk > 0:
                probs = model.predict_proba(x)[0]
                class_probs = sorted(zip(model.classes_, probs), key=lambda t: t[1], reverse=True)[:topk]
                # Store top-k as readable strings
                for k_i, (label, pval) in enumerate(class_probs, start=1):
                    out[f"top{k_i}_crate"] = label
                    out[f"top{k_i}_prob"] = float(pval)

            if write_probs and topk <= 0:
                # If user wants full probs, dump all classes (can be many columns)
                probs = model.predict_proba(x)[0]
                for label, pval in zip(model.classes_, probs):
                    out[f"prob__{label}"] = float(pval)

            rows.append(out)

        except Exception as e:
            failures.append({"title": title, "path": str(f), "error": f"{type(e).__name__}: {e}"})
            print(f"   ⚠️ Failed: {type(e).__name__}: {e}")

    return rows, failures


def write_csv_dicts(path: str, dict_rows: list[dict]):
    if not dict_rows:
        return
    cols = list(dict_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(dict_rows)


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Predict Serato crate(s) for new tracks using serato_model.pkl"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more audio files or folders containing audio files",
    )
    parser.add_argument(
        "--model",
        default="serato_model.pkl",
        help="Path to trained model .pkl (default: serato_model.pkl)",
    )
    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV for predictions (default: predictions.csv)",
    )
    parser.add_argument(
        "--failures",
        default="prediction_failures.csv",
        help="Output CSV for failures (default: prediction_failures.csv)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If an input is a folder, scan recursively",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Write top-k classes + probabilities (default: 3). Use 0 to disable.",
    )
    parser.add_argument(
        "--full-probs",
        action="store_true",
        help="Write probability column for EVERY class (can create many columns).",
    )

    args = parser.parse_args()

    model_path = resolve_to_existing_file(args.model)
    if model_path is None:
        print(f"❌ Model not found: {args.model}")
        print("   Make sure Phase 3 produced serato_model.pkl and you are in the correct folder.")
        sys.exit(1)

    bundle = joblib.load(str(model_path))
    if "model" not in bundle or "feature_columns" not in bundle:
        print("❌ Model file is not in expected format.")
        print("   Expected keys: 'model', 'feature_columns'")
        sys.exit(1)

    files = collect_audio_files(args.inputs, recursive=args.recursive)
    if not files:
        print("❌ No audio files found in the provided inputs.")
        sys.exit(1)

    print(f"🎧 Found {len(files)} audio files to predict.")
    print("--------------------------------------------------")

    rows, failures = predict_for_files(
        model_bundle=bundle,
        files=files,
        topk=args.topk,
        write_probs=args.full_probs,
    )

    write_csv_dicts(args.out, rows)
    if failures:
        write_csv_dicts(args.failures, failures)

    print("--------------------------------------------------")
    print(f"✅ Saved predictions to: {args.out}")
    if failures:
        print(f"⚠️ {len(failures)} files failed. See: {args.failures}")


if __name__ == "__main__":
    main()
