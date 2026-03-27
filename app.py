import streamlit as st
import pandas as pd
from pathlib import Path

from phase4_engine import collect_audio_files, load_model_bundle, propose_crates_for_files


st.set_page_config(page_title="SeratoAI", layout="wide")
st.title("🎧 SeratoAI — Predict Crates (Approve Before Writing)")

# -------------------------
# Session state
# -------------------------
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "fails_df" not in st.session_state:
    st.session_state.fails_df = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Model file", value="serato_model.pkl")
scan_path = st.sidebar.text_input(
    "Folder of new/unsorted music",
    value=str(Path.home() / "Downloads" / "Music")
)
recursive = st.sidebar.checkbox("Scan folders recursively", value=True)

st.sidebar.header("Prediction settings")
topk = st.sidebar.slider("Store Top-K alternatives (for later dropdown)", 1, 10, 3)

st.sidebar.divider()
st.sidebar.header("Serato write settings")
serato_root = st.sidebar.text_input(
    "Serato folder (root)",
    value=str(Path.home() / "Music" / "Serato")
)
dry_run = st.sidebar.checkbox("Dry run (no changes)", value=True)
make_backup = st.sidebar.checkbox("Backup before writing", value=True)

# -------------------------
# Buttons
# -------------------------
colA, colB = st.columns([1, 1])
with colA:
    run_predict = st.button("🔮 Generate Clean Predictions", type="primary")
with colB:
    apply_changes = st.button("✅ Apply APPROVED to Serato")

# -------------------------
# Predict
# -------------------------
if run_predict:
    try:
        bundle = load_model_bundle(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    files = collect_audio_files([scan_path], recursive=recursive)
    if not files:
        st.warning("No audio files found. Check the folder path and try again.")
        st.stop()

    pred_df, fails_df = propose_crates_for_files(bundle, files, topk=topk)

    if pred_df.empty:
        st.warning("No predictions produced. Check failures below.")
    else:
        # Add approval + final crate (for later dropdown improvements)
        pred_df["Approve"] = False
        pred_df["Final Crate"] = pred_df["Suggested Crate"]

        # Clean up values
        pred_df["Artist"] = pred_df["Artist"].fillna("").astype(str)
        pred_df["Genre"] = pred_df["Genre"].fillna("").astype(str)
        pred_df["Year"] = pred_df["Year"].fillna("").astype(str)

        # Round BPM + confidence for readability
        pred_df["BPM"] = pred_df["BPM"].astype(float).round(2)
        pred_df["Confidence"] = pred_df["Confidence"].astype(float).round(2)

        # Put columns in a nicer order
        display_order = [
            "Approve",
            "Song Title",
            "Artist",
            "BPM",
            "Genre",
            "Year",
            "Suggested Crate",
            "Confidence",
            "Final Crate",
            "path",  # hidden in editor
        ]
        # Keep any extra cols (like _top1_...) at the end
        extra_cols = [c for c in pred_df.columns if c not in display_order]
        pred_df = pred_df[display_order + extra_cols]

    st.session_state.pred_df = pred_df
    st.session_state.fails_df = fails_df

# -------------------------
# Display Predictions (Clean)
# -------------------------
pred_df = st.session_state.pred_df
fails_df = st.session_state.fails_df

if pred_df is not None and not pred_df.empty:
    st.subheader("Predictions (Approve / Deny)")
    st.caption("✅ Check Approve for tracks you want to add to Serato crates. Unchecked = Deny. Nothing writes until you click Apply.")

    # Only show clean columns in UI
    show_cols = [
        "Approve", "Song Title", "Artist", "BPM", "Genre", "Year",
        "Suggested Crate", "Confidence", "Final Crate"
    ]

    edited = st.data_editor(
        pred_df[show_cols + ["path"]],  # keep path for commit, but hide below
        use_container_width=True,
        hide_index=True,
        column_config={
            "Approve": st.column_config.CheckboxColumn("Approve"),
            "BPM": st.column_config.NumberColumn("BPM", format="%.2f"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            # For now Final Crate is editable as text; later we’ll make it a Selectbox dropdown
        },
        disabled=[
            "Song Title", "Artist", "BPM", "Genre", "Year",
            "Suggested Crate", "Confidence", "path"
        ],
    )

    # Merge edited columns back into full df (so we keep _topk columns for later)
    full = pred_df.copy()
    full.loc[:, "Approve"] = edited["Approve"].values
    full.loc[:, "Final Crate"] = edited["Final Crate"].values
    st.session_state.pred_df = full

    # Downloads
    st.download_button(
        "⬇️ Download clean_predictions.csv",
        data=full.drop(columns=[c for c in full.columns if c.startswith("_top")], errors="ignore")
                .to_csv(index=False).encode("utf-8"),
        file_name="clean_predictions.csv",
    )

if fails_df is not None and not fails_df.empty:
    st.subheader("Failures")
    st.dataframe(fails_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download failures.csv",
        data=fails_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_failures.csv",
    )

# -------------------------
# Apply Approved to Serato
# -------------------------
if apply_changes:
    full = st.session_state.pred_df
    if full is None or full.empty:
        st.warning("Generate predictions first.")
        st.stop()

    approved = full[full["Approve"] == True].copy()
    if approved.empty:
        st.info("No tracks approved. Nothing to apply.")
        st.stop()

    assignments = list(zip(approved["Final Crate"].astype(str), approved["path"].astype(str)))

    # ✅ Lazy import so the app can run even if pyserato import is flaky
    try:
        from serato_writer import write_tracks_to_crates
    except Exception as e:
        st.error(f"Serato writer failed to import: {e}")
        st.stop()

    try:
        results = write_tracks_to_crates(
            serato_root=Path(serato_root),
            assignments=assignments,
            dry_run=dry_run,
            make_backup=make_backup,
        )
    except Exception as e:
        st.error(f"Serato write failed: {e}")
        st.stop()

    res_df = pd.DataFrame([r.__dict__ for r in results])
    st.subheader("Serato Write Results")
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    if dry_run:
        st.info("Dry run is ON — no Serato files were modified.")
    else:
        st.success("Applied approved tracks to Serato crates (if success=True).")
