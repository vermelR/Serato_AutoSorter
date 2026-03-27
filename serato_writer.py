from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# --- Robust pyserato imports across versions ---
# Builder is NOT top-level in pyserato 0.2.1, so we import from pyserato.builder
try:
    from pyserato.builder import Builder  # ✅ works on 0.2.1
except Exception as e:
    raise ImportError(
        "Could not import Builder. Make sure pyserato is installed in this venv: "
        "python -m pip install pyserato"
    ) from e

# Crate location can vary across pyserato versions
try:
    from pyserato.crate import Crate  # common
except Exception:
    try:
        # Some versions place it differently
        from pyserato.model.crate import Crate  # fallback
    except Exception as e:
        raise ImportError(
            "Could not import Crate from pyserato. Your pyserato version may not support writing crates."
        ) from e


@dataclass
class SeratoWriteResult:
    crate_name: str
    track_path: str
    success: bool
    error: str = ""


def backup_serato_folder(serato_root: Path, backup_dir: Optional[Path] = None) -> Path:
    serato_root = serato_root.expanduser().resolve()
    if not serato_root.exists():
        raise FileNotFoundError(f"Serato folder not found: {serato_root}")

    if backup_dir is None:
        backup_dir = serato_root.parent / "SeratoAI_Backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{serato_root.name}_backup_{stamp}"
    shutil.copytree(serato_root, dest)
    return dest


def _save_crate(builder: Builder, crate: Crate, serato_root: Path) -> None:
    """
    pyserato Builder.save signature can vary; try common patterns.
    """
    serato_root = serato_root.expanduser().resolve()

    # Common kwarg names
    for kwargs in (
        {"root": serato_root},
        {"root_path": serato_root},
        {"path": serato_root},
    ):
        try:
            builder.save(crate, **kwargs)  # type: ignore
            return
        except TypeError:
            continue

    # Fallback (may write to default location)
    builder.save(crate)


def write_tracks_to_crates(
    serato_root: Path,
    assignments: Iterable[tuple[str, str]],
    dry_run: bool = True,
    make_backup: bool = True,
) -> list[SeratoWriteResult]:
    """
    assignments: iterable of (crate_name, track_path)
    """
    serato_root = serato_root.expanduser().resolve()

    if make_backup and not dry_run:
        backup = backup_serato_folder(serato_root)
        print(f"✅ Serato backup created: {backup}")

    results: list[SeratoWriteResult] = []
    builder = Builder()

    for crate_name, track_path in assignments:
        crate_name = str(crate_name).strip()
        track_path = str(track_path).strip()

        try:
            if not crate_name:
                results.append(SeratoWriteResult(crate_name, track_path, False, "Empty crate name"))
                continue

            p = Path(track_path).expanduser().resolve()
            if not p.exists():
                results.append(SeratoWriteResult(crate_name, str(p), False, "Track file not found"))
                continue

            if dry_run:
                results.append(SeratoWriteResult(crate_name, str(p), True, "DRY_RUN"))
                continue

            crate = Crate(crate_name)
            crate.add_track(str(p))

            _save_crate(builder, crate, serato_root)

            results.append(SeratoWriteResult(crate_name, str(p), True))

        except Exception as e:
            results.append(SeratoWriteResult(crate_name, track_path, False, f"{type(e).__name__}: {e}"))

    return results
