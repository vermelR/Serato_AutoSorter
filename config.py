"""
Central configuration for SeratoAI.

Most values can be overridden with environment variables so nothing
sensitive (like API keys) needs to be hardcoded or committed.
"""

import os
from pathlib import Path

# -------------------------
# Models
# -------------------------
CRATE_MODEL_PATH = os.environ.get("SERATO_CRATE_MODEL", "serato_model.pkl")
GENRE_MODEL_PATH = os.environ.get("SERATO_GENRE_MODEL", "serato_genre_model.pkl")

# -------------------------
# AcoustID / MusicBrainz (audio fingerprint identification)
# -------------------------
# Free API key from https://acoustid.org/api-key
ACOUSTID_API_KEY = os.environ.get("ACOUSTID_API_KEY", "")

# MusicBrainz requires a real contact string in the User-Agent for its
# public API. Override with MB_CONTACT_EMAIL if you want to use a
# different address.
MB_APP_NAME = "SeratoAI"
MB_APP_VERSION = "0.2"
MB_CONTACT_EMAIL = os.environ.get("MB_CONTACT_EMAIL", "rohandesai675@gmail.com")

# Minimum AcoustID match score (0-1) before we trust the fingerprint result
ACOUSTID_MIN_SCORE = float(os.environ.get("ACOUSTID_MIN_SCORE", "0.6"))

# Minimum confidence (0-1) before we trust the local ML genre classifier
GENRE_MODEL_MIN_CONFIDENCE = float(os.environ.get("GENRE_MODEL_MIN_CONFIDENCE", "0.4"))

# -------------------------
# Watcher (auto-detect new music)
# -------------------------
# Folders to watch for newly added audio files (Downloads, an "Inbox"/"New"
# folder, etc). Comma-separated in the env var, one path per line in the UI.
DEFAULT_WATCH_FOLDERS = [
    str(Path.home() / "Downloads" / "Music"),
]

# Serato ".crate" files to treat as an inbox: any track path listed inside
# these crates is treated as a "new" track waiting to be sorted. Useful if
# you drag new tracks into a "New"/"Unsorted" crate in Serato.
DEFAULT_WATCH_CRATES: list[str] = []

# How long (seconds) a file's size must stay unchanged before we consider it
# "fully downloaded" and safe to analyze.
FILE_STABLE_SECONDS = float(os.environ.get("SERATO_FILE_STABLE_SECONDS", "3"))

# How often (seconds) the watcher re-scans watched crates for new entries.
CRATE_POLL_SECONDS = float(os.environ.get("SERATO_CRATE_POLL_SECONDS", "15"))

# Where discovered-but-not-yet-reviewed tracks are queued for the app to pick up.
PENDING_QUEUE_PATH = os.environ.get("SERATO_PENDING_QUEUE", "pending_review.jsonl")

# Where we remember which files have already been queued, so the watcher
# doesn't re-propose the same track every time it restarts.
PROCESSED_INDEX_PATH = os.environ.get("SERATO_PROCESSED_INDEX", "processed_index.json")
