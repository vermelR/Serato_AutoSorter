import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "music_features_dataset.csv"
MODEL_FILE = "serato_model.pkl"

NON_FEATURE_COLS = ["title", "path", "crate"]

def main():
    df = pd.read_csv(DATA_FILE)

    # Clean label
    df["crate"] = df["crate"].astype(str).str.strip()
    df = df[df["crate"].notna() & (df["crate"] != "")].copy()

    # Build X/y
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["crate"]

    # Force numeric features
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Remove crates with too few samples (helps stratify & evaluation)
    counts = y.value_counts()
    keep = counts[counts >= 2].index
    X = X[y.isin(keep)]
    y = y[y.isin(keep)]

    if y.nunique() < 2:
        print("❌ Need at least 2 crate classes with >=2 samples each to train/evaluate.")
        print("Crate counts:\n", counts.head(20))
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🧠 Training on {len(X_train)} tracks | {y.nunique()} classes")
    print(f"🎛️ Features: {list(X.columns)}")

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n✅ Accuracy: {acc*100:.2f}%")
    print("\n--- Classification report ---")
    print(classification_report(y_test, preds, zero_division=0))

    # Save model + feature list (so prediction uses same column order)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODEL_FILE)
    print(f"\n💾 Saved: {MODEL_FILE}")

if __name__ == "__main__":
    main()
