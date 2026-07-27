"""Read-only diagnostics for duplicate-looking Serato tracks.

Run this module against a Serato root before manually cleaning any library
records.  It does not mutate crate files, ``database V2``, or audio files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from serato_crate import (
    SeratoCrateParseError,
    canonical_track_path,
    parse_serato_crate,
    parse_serato_database_paths,
)


def _path_record(path: str, serialized_path: str, **extra: str) -> dict[str, str]:
    canonical_path, comparison_key = canonical_track_path(path)
    return {
        "serialized_path": serialized_path,
        "canonical_path": str(canonical_path),
        "comparison_key": comparison_key,
        **extra,
    }


def _duplicates(records: list[dict[str, str]], scope_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record[scope_key], record["comparison_key"])].append(record)
    return [
        {
            scope_key: scope,
            "comparison_key": path_key,
            "records": values,
        }
        for (scope, path_key), values in grouped.items()
        if len(values) > 1
    ]


def diagnose_serato_duplicates(serato_root: str | Path) -> dict[str, Any]:
    """Return raw crate/database path representations without changing them."""
    root = Path(serato_root).expanduser().resolve()
    subcrates = next(
        (root / name for name in ("SubCrates", "Subcrates") if (root / name).is_dir()),
        root / "SubCrates",
    )
    crate_records: list[dict[str, str]] = []
    crate_errors: list[dict[str, str]] = []

    if subcrates.is_dir():
        for crate_file in sorted(subcrates.glob("*.crate")):
            try:
                for entry in parse_serato_crate(crate_file):
                    if entry.path:
                        crate_records.append(
                            _path_record(
                                entry.path,
                                entry.serialized_path,
                                crate_file=str(crate_file),
                                crate_name=crate_file.stem,
                            )
                        )
            except SeratoCrateParseError as exc:
                crate_errors.append({"crate_file": str(crate_file), "error": str(exc)})

    references_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in crate_records:
        references_by_path[record["comparison_key"]].append(record)
    multi_crate_references = [
        {
            "comparison_key": path_key,
            "records": records,
        }
        for path_key, records in references_by_path.items()
        if len({record["crate_file"] for record in records}) > 1
    ]

    database_file = root / "database V2"
    database_records: list[dict[str, str]] = []
    database_error = ""
    if database_file.is_file():
        try:
            for entry in parse_serato_database_paths(database_file):
                if entry.path:
                    database_records.append(
                        _path_record(
                            entry.path,
                            entry.serialized_path,
                            byte_range=f"{entry.field_start}:{entry.field_end}",
                        )
                    )
        except SeratoCrateParseError as exc:
            database_error = str(exc)

    return {
        "serato_root": str(root),
        "crate_records": crate_records,
        "crate_parse_errors": crate_errors,
        "duplicate_entries_in_one_crate": _duplicates(crate_records, "crate_file"),
        "references_in_multiple_crates": multi_crate_references,
        "database_v2": {
            "path": str(database_file),
            "exists": database_file.is_file(),
            "records": database_records,
            "duplicate_records": _duplicates(database_records, "comparison_key"),
            "error": database_error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Serato duplicate diagnostic")
    parser.add_argument("serato_root", help="Path to the _Serato_ folder")
    args = parser.parse_args()
    print(json.dumps(diagnose_serato_duplicates(args.serato_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
