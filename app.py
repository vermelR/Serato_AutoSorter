import streamlit as st
import pandas as pd
from pathlib import Path

import config
from phase4_engine import collect_audio_files, load_model_bundle, propose_crates_for_files
from tag_writer import write_genre_year
from watcher import Watcher, load_pending_queue, mark_reviewed


st.set_page_config(page_title="SeratoAI", layout="wide")
st.title("🎧 SeratoAI — Predict Crates, Genre & Year (Approve Before Writing)")

# -------------------------
# Session state
# -------------------------
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "fails_df" not in st.session_state:
    st.session_state.fails_df = None
if "watcher" not in st.session_state:
    st.session_state.watcher = None
if "queue_df" not in st.session_state:
    st.session_state.queue_df = None

SHOW_COLS = [
    "Approve", "Song Title", "Artist", "BPM",
    "Genre", "Genre Source", "Year", "Year Source",
    "Suggested Crate", "Confidence", "Final Crate",
]
EDITABLE_COLS = {"Approve", "Genre", "Year", "Final Crate"}


def apply_approved(approved: pd.DataFrame, serato_root: str, dry_run: bool, make_backup: bool):
    """Write genre/year tags into each approved file, then assign it to its
    crate in Serato. Returns (tag_results_df, crate_results_df)."""
    tag_results = []
    for _, r in approved.iterrows():
        if dry_run:
            tag_results.append({"path": r["path"], "success": True, "error": "DRY_RUN"})
        else:
            res = write_genre_year(str(r["path"]), genre=str(r.get("Genre", "")), year=str(r.get("Year", "")))
            tag_results.append({"path": res.path, "success": res.success, "error": res.error})

    assignments = list(zip(approved["Final Crate"].astype(str), approved["path"].astype(str)))

    try:
        from serato_writer import write_tracks_to_crates
    except Exception as e:
        st.error(f"Serato writer failed to import: {e}")
        return pd.DataFrame(tag_results), pd.DataFrame()

    try:
        crate_results = write_tracks_to_crates(
            serato_root=Path(serato_root),
            assignments=assignments,
            dry_run=dry_run,
            make_backup=make_backup,
        )
    except Exception as e:
        st.error(f"Serato write failed: {e}")
        return pd.DataFrame(tag_results), pd.DataFrame()

    return pd.DataFrame(tag_results), pd.DataFrame([r.__dict__ for r in crate_results])


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Crate model file", value=config.CRATE_MODEL_PATH)
scan_path = st.sidebar.text_input(
    "Folder of new/unsorted music (manual scan)",
    value=str(Path.home() / "Downloads" / "Music")
)
recursive = st.sidebar.checkbox("Scan folders recursively", value=True)

st.sidebar.header("Prediction settings")
topk = st.sidebar.slider("Store Top-K alternatives (for later dropdown)", 1, 10, 3)

st.sidebar.divider()
st.sidebar.header("Genre & Year identification")
identify_enabled = st.sidebar.checkbox(
    "Identify genre & year (audio fingerprint + local ML fallback)", value=True,
    help="Fingerprints each track and looks up its real genre/year via AcoustID + "
         "MusicBrainz. Falls back to a local ML genre classifier, then the file's "
         "existing tag, if no confident match is found."
)
acoustid_key_input = st.sidebar.text_input(
    "AcoustID API key", value=config.ACOUSTID_API_KEY, type="password",
    help="Free key from https://acoustid.org/api-key. Leave blank to skip "
         "fingerprint identification and use the local ML/tag fallback only."
)
if acoustid_key_input.strip():
    config.ACOUSTID_API_KEY = acoustid_key_input.strip()

st.sidebar.divider()
st.sidebar.header("Serato write settings")
serato_root = st.sidebar.text_input(
    "Serato folder (root)",
    value=str(Path.home() / "Music" / "Serato")
)
dry_run = st.sidebar.checkbox("Dry run (no changes)", value=True)
make_backup = st.sidebar.checkbox("Backup before writing", value=True)

st.sidebar.divider()
st.sidebar.header("Background watcher")
st.sidebar.caption("Automatically watches folders (and/or Serato inbox crates) for new "
                    "tracks and runs predictions in the background. Nothing is written "
                    "to Serato/your files until you approve it below.")
watch_folders_text = st.sidebar.text_area(
    "Watch folders (one per line)",
    value="\n".join(config.DEFAULT_WATCH_FOLDERS),
)
watch_crates_text = st.sidebar.text_area(
    "Watch Serato inbox crate files (.crate, one per line, optional)",
    value="",
    help="e.g. a 'New'/'Unsorted' crate you drag new tracks into inside Serato.",
)

colW1, colW2 = st.sidebar.columns(2)
start_watch = colW1.button("▶ Start watching")
stop_watch = colW2.button("⏹ Stop watching")

if start_watch:
    folders = [l.strip() for l in watch_folders_text.splitlines() if l.strip()]
    crates = [l.strip() for l in watch_crates_text.splitlines() if l.strip()]
    if st.session_state.watcher is not None:
        st.session_state.watcher.stop()
    st.session_state.watcher = Watcher(
        folders=folders, crates=crates, model_path=model_path, topk=topk,
        identify_genre=identify_enabled,
    )
    st.session_state.watcher.start()

if stop_watch and st.session_state.watcher is not None:
    st.session_state.watcher.stop()

if st.session_state.watcher is not None:
    status = st.session_state.watcher.status()
    state = "🟢 running" if status["running"] else "🔴 stopped"
    st.sidebar.caption(f"Watcher: {state} | queued: {status['queued']}")
    if status["last_error"]:
        st.sidebar.caption(f"⚠️ {status['last_error']}")
else:
    st.sidebar.caption("Watcher: 🔴 not started")

# -------------------------
# Buttons
# -------------------------
colA, colB = st.columns([1, 1])
with colA:
    run_predict = st.button("🔮 Generate Predictions (manual scan)", type="primary")
with colB:
    apply_changes = st.button("✅ Apply APPROVED to Serato")

# -------------------------
# Predict (manual scan)
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

    pred_df, fails_df = propose_crates_for_files(bundle, files, topk=topk, identify_genre=identify_enabled)

    if pred_df.empty:
        st.warning("No predictions produced. Check failures below.")
    else:
        pred_df["Approve"] = False
        pred_df["Final Crate"] = pred_df["Suggested Crate"]

        pred_df["Artist"] = pred_df["Artist"].fillna("").astype(str)
        pred_df["Genre"] = pred_df["Genre"].fillna("").astype(str)
        pred_df["Year"] = pred_df["Year"].fillna("").astype(str)

        pred_df["BPM"] = pred_df["BPM"].astype(float).round(2)
        pred_df["Confidence"] = pred_df["Confidence"].astype(float).round(2)

        display_order = SHOW_COLS + ["path"]
        extra_cols = [c for c in pred_df.columns if c not in display_order]
        pred_df = pred_df[display_order + extra_cols]

    st.session_state.pred_df = pred_df
    st.session_state.fails_df = fails_df

# -------------------------
# Display Predictions (manual scan)
# -------------------------
pred_df = st.session_state.pred_df
fails_df = st.session_state.fails_df

if pred_df is not None and not pred_df.empty:
    st.subheader("Manual Scan — Predictions (Approve / Deny)")
    st.caption("✅ Check Approve for tracks you want to add to Serato crates (with genre/year "
               "written into the file). Unchecked = Deny. Nothing writes until you click Apply.")

    edited = st.data_editor(
        pred_df[SHOW_COLS + ["path"]],
        use_container_width=True,
        hide_index=True,
        key="manual_editor",
        column_config={
            "Approve": st.column_config.CheckboxColumn("Approve"),
            "BPM": st.column_config.NumberColumn("BPM", format="%.2f"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
        disabled=[c for c in SHOW_COLS if c not in EDITABLE_COLS] + ["path"],
    )

    full = pred_df.copy()
    for col in EDITABLE_COLS:
        full.loc[:, col] = edited[col].values
    st.session_state.pred_df = full

    st.download_button(
        "⬇️ Download clean_predictions.csv",
        data=full.drop(columns=[c for c in full.columns if c.startswith("_top")], errors="ignore")
                .to_csv(index=False).encode("utf-8"),
        file_name="clean_predictions.csv",
    )

if fails_df is not None and not fails_df.empty:
    st.subheader("Manual Scan — Failures")
    st.dataframe(fails_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download failures.csv",
        data=fails_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_failures.csv",
    )

# -------------------------
# Auto-detected queue (from background watcher)
# -------------------------
st.divider()
st.subheader("🔎 Auto-Detected (from watched folders / inbox crates)")

refresh_queue = st.button("🔄 Refresh auto-detected queue")
queue_rows = load_pending_queue()

if queue_rows:
    queue_df = pd.DataFrame(queue_rows)
    queue_df["Approve"] = False
    if "Final Crate" not in queue_df.columns:
        queue_df["Final Crate"] = queue_df["Suggested Crate"]
    for col in ("Artist", "Genre", "Genre Source", "Year", "Year Source"):
        if col not in queue_df.columns:
            queue_df[col] = ""
        queue_df[col] = queue_df[col].fillna("").astype(str)
    queue_df["BPM"] = queue_df["BPM"].astype(float).round(2)
    queue_df["Confidence"] = queue_df["Confidence"].astype(float).round(2)

    st.caption(f"{len(queue_df)} track(s) auto-detected and waiting for review.")

    queue_edited = st.data_editor(
        queue_df[SHOW_COLS + ["path"]],
        use_container_width=True,
        hide_index=True,
        key="queue_editor",
        column_config={
            "Approve": st.column_config.CheckboxColumn("Approve"),
            "BPM": st.column_config.NumberColumn("BPM", format="%.2f"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
        disabled=[c for c in SHOW_COLS if c not in EDITABLE_COLS] + ["path"],
    )

    queue_full = queue_df.copy()
    for col in EDITABLE_COLS:
        queue_full.loc[:, col] = queue_edited[col].values
    st.session_state.queue_df = queue_full

    apply_queue = st.button("✅ Apply reviewed auto-detected tracks")

    if apply_queue:
        approved = queue_full[queue_full["Approve"] == True].copy()
        all_paths = set(queue_full["path"].astype(str))

        if not approved.empty:
            tag_res_df, crate_res_df = apply_approved(approved, serato_root, dry_run, make_backup)
            st.subheader("Auto-Detected — Tag Write Results")
            st.dataframe(tag_res_df, use_container_width=True, hide_index=True)
            st.subheader("Auto-Detected — Serato Write Results")
            st.dataframe(crate_res_df, use_container_width=True, hide_index=True)

        if not dry_run:
            # Remove every reviewed row (approved or denied) from the queue so
            # it isn't re-suggested; denied tracks are remembered as "seen".
            if st.session_state.watcher is not None:
                st.session_state.watcher.mark_processed(all_paths)
            else:
                mark_reviewed(all_paths)
            st.success("Applied. Reviewed tracks removed from the auto-detected queue.")
        else:
            st.info("Dry run is ON — queue left untouched so you can re-review with dry run off.")
else:
    st.caption("Nothing queued yet. Start the watcher in the sidebar and drop new tracks "
               "into a watched folder (or inbox crate).")

# -------------------------
# Apply Approved to Serato (manual scan)
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

    tag_res_df, crate_res_df = apply_approved(approved, serato_root, dry_run, make_backup)

    st.subheader("Manual Scan — Tag Write Results")
    st.dataframe(tag_res_df, use_container_width=True, hide_index=True)
    st.subheader("Manual Scan — Serato Write Results")
    st.dataframe(crate_res_df, use_container_width=True, hide_index=True)

    if dry_run:
        st.info("Dry run is ON — no Serato files or tags were modified.")
    else:
        st.success("Applied approved tracks to Serato crates and wrote genre/year tags (if success=True).")
