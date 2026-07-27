"""
Local, offline genre classifier used only when a track has no valid embedded
Genre tag. It never performs metadata lookup and never predicts Year.

Trained the same way as the crate classifier (phase3_train.py): scan a
folder of already-tagged music, extract the same audio features used
elsewhere in the app (bpm/brightness/energy/mfccs), and fit a
RandomForestClassifier against each track's existing genre tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config

GENRE_TRAINING_CSV = "genre_training_data.csv"


def harvest_genre_training_data(folders: list[str], recursive: bool = True) -> pd.DataFrame:
    """Scan folders for audio files that already have a genre tag, extract
    audio features, and return a DataFrame ready for training."""
    # Imported lazily to avoid a circular import (phase4_engine imports this
    # module for its identify_genre_year() fallback).
    from phase4_engine import collect_audio_files, extract_audio_features, read_track_metadata

    files = collect_audio_files(folders, recursive=recursive)

    rows = []
    for idx, f in enumerate(files, start=1):
        meta = read_track_metadata(f)
        genre = meta.get("genre", "").strip()
        if not genre:
            continue
        try:
            feats = extract_audio_features(str(f))
        except Exception:
            continue
        rows.append([meta["title"], str(f), genre] + feats)
        print(f"[{idx}/{len(files)}] {meta['title']} -> genre={genre}")

    columns = ["title", "path", "genre", "bpm", "brightness", "energy"] + [
        f"mfcc_{j}" for j in range(1, 14)
    ]
    return pd.DataFrame(rows, columns=columns)


def train_genre_model(
    training_csv: str = GENRE_TRAINING_CSV,
    model_out: str = config.GENRE_MODEL_PATH,
) -> Optional[dict]:
    df = pd.read_csv(training_csv)
    df["genre"] = df["genre"].astype(str).str.strip()
    df = df[df["genre"].notna() & (df["genre"] != "")].copy()

    drop_cols = [c for c in ("title", "path", "genre") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["genre"]

    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    counts = y.value_counts()
    keep = counts[counts >= 2].index
    X = X[y.isin(keep)]
    y = y[y.isin(keep)]

    if y.nunique() < 2:
        print("Need at least 2 genre classes with >=2 samples each to train.")
        print("Genre counts:\n", counts.head(20))
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Genre model accuracy: {acc * 100:.2f}%  ({y.nunique()} classes)")

    bundle = {"model": model, "feature_columns": list(X.columns)}
    joblib.dump(bundle, model_out)
    print(f"Saved: {model_out}")
    return bundle


_genre_bundle_cache: dict | None = None


def load_genre_model(model_path: str = config.GENRE_MODEL_PATH) -> Optional[dict]:
    global _genre_bundle_cache
    if _genre_bundle_cache is not None:
        return _genre_bundle_cache
    p = Path(model_path)
    if not p.exists():
        return None
    try:
        _genre_bundle_cache = joblib.load(str(p))
    except Exception:
        return None
    return _genre_bundle_cache


def predict_genre(feats: list[float], model_path: str = config.GENRE_MODEL_PATH):
    """Returns (genre_label, confidence) or (None, 0.0) if no model/feature mismatch."""
    bundle = load_genre_model(model_path)
    if not bundle:
        return None, 0.0

    feature_cols = bundle["feature_columns"]
    model = bundle["model"]

    x = pd.DataFrame([feats[: len(feature_cols)]], columns=feature_cols)
    probs = model.predict_proba(x)[0]
    best_idx = int(np.argmax(probs))
    return str(model.classes_[best_idx]), float(probs[best_idx])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the local fallback genre classifier")
    parser.add_argument("folders", nargs="+", help="Folders of already-genre-tagged music to learn from")
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    df = harvest_genre_training_data(args.folders, recursive=args.recursive)
    if df.empty:
        print("No genre-tagged tracks found to train on.")
    else:
        df.to_csv(GENRE_TRAINING_CSV, index=False)
        print(f"Saved {len(df)} training rows to {GENRE_TRAINING_CSV}")
        train_genre_model()
