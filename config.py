"""Backward-compatible configuration constants sourced from typed settings.

New application code should pass :class:`serato_ai.settings.ApplicationSettings`
where practical. These mutable module values remain for existing integrations
and tests that intentionally override runtime configuration.
"""

from serato_ai.settings import load_settings


_settings = load_settings()

CRATE_MODEL_PATH = _settings.crate_model_path
GENRE_MODEL_PATH = _settings.genre_model_path
ACOUSTID_API_KEY = _settings.acoustid_api_key
MB_APP_NAME = _settings.mb_app_name
MB_APP_VERSION = _settings.mb_app_version
MB_CONTACT_EMAIL = _settings.mb_contact_email
ACOUSTID_MIN_SCORE = _settings.acoustid_min_score
GENRE_MODEL_MIN_CONFIDENCE = _settings.genre_model_min_confidence
DEFAULT_WATCH_FOLDERS = list(_settings.default_watch_folders)
DEFAULT_WATCH_CRATES: list[str] = list(_settings.default_watch_crates)
FILE_STABLE_SECONDS = _settings.file_stable_seconds
CRATE_POLL_SECONDS = _settings.crate_poll_seconds
PENDING_QUEUE_PATH = _settings.pending_queue_path
PROCESSED_INDEX_PATH = _settings.processed_index_path
