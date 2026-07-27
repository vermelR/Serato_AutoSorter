from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import librosa
import numpy as np
import pandas as pd
from mutagen import File as MutagenFile

import config
from serato_ai.core.confidence import filter_and_rank_independent_probabilities, filter_and_rank_probabilities
from serato_ai.core.models import EvaluationConfiguration, ThresholdConfiguration
from serato_ai.core.path_utils import normalize_path
from serato_ai.core.quality_rules import prediction_quality_state
from serato_ai.core.storage_rules import StorageBudgets, assess_model_size
from serato_ai.infrastructure.metadata_providers import EmbeddedTagProvider
from serato_ai.services.metadata_enrichment_service import MetadataEnrichmentService
from serato_ai.settings import load_settings

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
        tempo = float(np.asarray(tempo).reshape(-1)[0])

        # 2) Brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(centroid))

        # 3) Energy
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # 4) MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0).astype(float).tolist()

        return [tempo, brightness, energy] + mfccs_scaled

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
    settings = load_settings()
    assessment = assess_model_size(
        p.stat().st_size,
        StorageBudgets(
            preferred_bytes=settings.preferred_model_bytes,
            warning_bytes=settings.model_warning_bytes,
            review_bytes=settings.model_review_bytes,
            hard_limit_bytes=settings.model_hard_limit_bytes,
            allow_developer_override=settings.allow_oversized_model,
        ),
    )
    if not assessment.automatic_activation_allowed:
        raise ValueError(
            "Model is too large to load safely. "
            + " ".join(assessment.warnings)
            + " Build a compact candidate; the legacy file has not been changed."
        )
    bundle = joblib.load(str(p))
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model bundle missing keys: 'model' and/or 'feature_columns'")
    model = bundle["model"]
    if not hasattr(model, "classes_"):
        raise ValueError("Model bundle model is missing classes_")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model bundle model is missing predict_proba")
    if not isinstance(bundle["feature_columns"], (list, tuple)):
        raise ValueError("Model bundle feature_columns must be a list or tuple")
    bundle.setdefault("bundle_schema_version", "legacy")
    bundle.setdefault("model_version", p.stem)
    bundle.setdefault("artifact_size_bytes", p.stat().st_size)
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
# Offline-first Genre / Year identification
# -------------------------
def _offline_metadata_service() -> MetadataEnrichmentService:
    """Build the shared tags-first service without enabling online providers."""
    return MetadataEnrichmentService(
        embedded_provider=EmbeddedTagProvider(read_track_metadata),
        minimum_local_confidence=config.GENRE_MODEL_MIN_CONFIDENCE,
    )


def identify_genre_year(path: Path, feats: list[float], existing_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for the offline-first metadata contract.

    Existing Genre and Year tags are always considered first. Local ML can fill
    only a missing Genre; missing/invalid Year remains blank. No fingerprint or
    provider lookup is attempted here.
    """
    service = MetadataEnrichmentService(
        embedded_provider=EmbeddedTagProvider(lambda _path: existing_meta),
        minimum_local_confidence=config.GENRE_MODEL_MIN_CONFIDENCE,
    )
    result = service.enrich(path, features=tuple(feats), use_local_genre=True)
    return {
        "genre": result.genre,
        "year": result.year,
        "genre_source": result.genre_source,
        "year_source": result.year_source,
        "genre_confidence": result.genre_confidence,
        "online_lookup_attempted": result.online_lookup_attempted,
        "provider_status": result.provider_status,
    }


def _prediction_quality_configuration(bundle: dict) -> tuple[EvaluationConfiguration, ThresholdConfiguration, dict[str, int]]:
    """Read persisted M5 quality settings, falling back safely for older models."""
    raw = bundle.get("quality_configuration", {})
    if not isinstance(raw, dict):
        raw = {}
    threshold_raw = raw.get("threshold_configuration", {})
    if not isinstance(threshold_raw, dict):
        threshold_raw = {}
    try:
        thresholds = ThresholdConfiguration(
            global_threshold=float(threshold_raw.get("global_threshold", 0.50)),
            per_crate=tuple((str(crate), float(value)) for crate, value in threshold_raw.get("per_crate", ())),
            minimum_threshold=float(threshold_raw.get("minimum_threshold", 0.20)),
            maximum_threshold=float(threshold_raw.get("maximum_threshold", 0.80)),
            minimum_support=int(threshold_raw.get("minimum_support", 3)),
            objective=str(threshold_raw.get("objective", "f1")),
            source_split=str(threshold_raw.get("source_split", "legacy_default")),
        )
    except (TypeError, ValueError):
        thresholds = ThresholdConfiguration()
    configuration = EvaluationConfiguration(
        low_confidence_probability=float(raw.get("low_confidence_probability", 0.40)),
        low_confidence_margin=float(raw.get("low_confidence_margin", 0.05)),
        per_crate_minimum_support=int(raw.get("per_crate_minimum_support", 2)),
    )
    support = raw.get("training_support", {})
    return configuration, thresholds, {str(crate): int(value) for crate, value in support.items()} if isinstance(support, dict) else {}


# -------------------------
# Prediction proposals (clean)
# -------------------------
def propose_crates_for_files(
    bundle: dict,
    files: list[Path],
    topk: int = 3,
    identify_genre: bool = True,
    excluded_crates: set[str] | None = None,
    allowed_crates: set[str] | None = None,
    metadata_service: MetadataEnrichmentService | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    rows = []
    fails = []
    metadata = metadata_service or _offline_metadata_service()
    quality_configuration, thresholds, support_by_crate = _prediction_quality_configuration(bundle)
    prepared: list[tuple[Path, list[float], Any]] = []

    for f in files:
        try:
            # This must happen before audio extraction, model prediction, or
            # any local fallback so valid tags are always authoritative.
            embedded = metadata.read_embedded(f)
            feats = extract_audio_features(str(f))
            enriched = metadata.enrich(
                f, features=tuple(feats), embedded=embedded, use_local_genre=identify_genre,
            )
            prepared.append((f, feats, enriched))
        except Exception as e:
            fails.append({
                "Song Title": f.stem,
                "path": str(f),
                "error": f"{type(e).__name__}: {e}",
            })

    ranker = (
        filter_and_rank_independent_probabilities
        if bundle.get("prediction_semantics") == "independent_multilabel"
        else filter_and_rank_probabilities
    )

    def append_prediction(f: Path, feats: list[float], enriched, probs) -> None:
        try:
            suggestions = ranker(
                zip(model.classes_, probs),
                allowed_crates=allowed_crates,
                excluded_crates=excluded_crates,
            )
            pred, conf = suggestions[0].crate_name, suggestions[0].probability
            quality = prediction_quality_state(
                tuple((suggestion.crate_name, suggestion.probability) for suggestion in suggestions),
                thresholds, quality_configuration, support_by_crate=support_by_crate,
            )

            row = {
                "Song Title": enriched.title or f.stem,
                "Artist": enriched.artist,
                "BPM": round(float(feats[0]), 2),
                "Genre": enriched.genre,
                "Genre Source": enriched.genre_source,
                "Genre Confidence": float(enriched.genre_confidence),
                "Year": enriched.year,
                "Year Source": enriched.year_source,
                "_metadata_raw_year": enriched.raw_year,
                "Suggested Crate": pred,
                # Keep the full conditional probability. The UI formats it as
                # a percentage, so rounding here would make displayed values
                # inaccurate after the allow-list is renormalized.
                "Confidence": float(conf),
                "Prediction Quality": quality.prediction_quality.replace("_", " ").title(),
                "Needs Review": quality.needs_review,
                "Review Reason": quality.review_reason,
                "Top Margin": float(quality.top_margin),
                "Threshold Used": float(quality.threshold_used),
                "Supported Crate Count": quality.supported_crate_count,
                "path": str(f),  # keep hidden for commit
                "_allowed_crates": sorted(suggestion.crate_name for suggestion in suggestions),
                "_metadata_provider_status": list(enriched.provider_status),
                "_metadata_warnings": list(enriched.warnings),
                "_online_lookup_attempted": enriched.online_lookup_attempted,
            }

            for suggestion in suggestions[:max(0, int(topk))]:
                row[f"_top{suggestion.rank}_crate"] = suggestion.crate_name
                row[f"_top{suggestion.rank}_prob"] = float(suggestion.probability)

            rows.append(row)
        except Exception as e:
            fails.append({
                "Song Title": f.stem,
                "path": str(f),
                "error": f"{type(e).__name__}: {e}",
            })

    batch_size = load_settings().prediction_batch_size
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        matrix = pd.DataFrame(
            np.asarray([features for _, features, _ in batch], dtype=np.float32),
            columns=feature_cols,
        )
        try:
            probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
            if probabilities.shape != (len(batch), len(model.classes_)):
                raise ValueError("Batch probability shape does not match tracks and classes.")
            for (path, features, enriched), probability_row in zip(batch, probabilities):
                append_prediction(path, features, enriched, probability_row)
        except Exception:
            # A single unreadable/corrupt row must not discard a healthy batch.
            # The fallback also preserves compatibility with older estimators
            # that only accept one row at a time.
            for path, features, enriched in batch:
                try:
                    single = pd.DataFrame(
                        np.asarray([features], dtype=np.float32),
                        columns=feature_cols,
                    )
                    probability_row = np.asarray(model.predict_proba(single), dtype=float)
                    if probability_row.shape != (1, len(model.classes_)):
                        raise ValueError("Probability shape does not match the model class list.")
                    append_prediction(path, features, enriched, probability_row[0])
                except Exception as exc:
                    fails.append({
                        "Song Title": path.stem,
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    return pd.DataFrame(rows), pd.DataFrame(fails)
