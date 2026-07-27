from __future__ import annotations

import tempfile
import unicodedata
import unittest
from pathlib import Path

import pytest

from phase1_harvest import parse_serato_crate
from serato_crate import (
    SeratoCrateParseError,
    canonical_track_path,
    parse_serato_crate as parse_serato_crate_entries,
    parse_serato_crate_bytes,
    serialize_serato_track_record,
)
from serato_diagnostics import diagnose_serato_duplicates
from serato_writer import write_tracks_to_crates


pytestmark = [pytest.mark.writer, pytest.mark.integration]


def _crate_header() -> bytes:
    # The parser requires Serato's binary marker.  Tests add only the records
    # under examination, avoiding any real Serato data.
    return b"vrsn\x00\x00\x00\x00"


def _record(payload: bytes) -> bytes:
    return b"otrk" + (len(payload) + 8).to_bytes(4, "big") + b"ptrk" + len(payload).to_bytes(4, "big") + payload


class SeratoCrateParserTests(unittest.TestCase):
    def test_empty_crate_file_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(SeratoCrateParseError, "crate file is empty"):
            parse_serato_crate_bytes(b"", "empty.crate")

    def test_one_byte_utf16_field_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(SeratoCrateParseError, "expected an even UTF-16 byte length, received 1 byte"):
            parse_serato_crate_bytes(_crate_header() + _record(b"/"), "one-byte.crate")

    def test_odd_length_utf16_field_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(SeratoCrateParseError, "expected an even UTF-16 byte length, received 3 bytes"):
            parse_serato_crate_bytes(_crate_header() + _record(b"\x00/\x00"), "odd-length.crate")

    def test_valid_utf16_le_path_field_is_decoded_without_error(self) -> None:
        expected = "/tmp/valid-le.mp3"
        entries = parse_serato_crate_bytes(
            _crate_header() + _record(expected.encode("utf-16-le")), "valid-le.crate"
        )
        self.assertEqual([entry.path for entry in entries], [expected])
        self.assertEqual(entries[0].serialized_path, expected)
        self.assertEqual(entries[0].encoding, "utf-16-le")

    def test_valid_serato_utf16_be_path_field_is_decoded_without_error(self) -> None:
        expected = "/tmp/valid-be.mp3"
        entries = parse_serato_crate_bytes(
            _crate_header() + serialize_serato_track_record(expected), "valid-be.crate"
        )
        self.assertEqual([entry.path for entry in entries], [expected])
        self.assertEqual(entries[0].serialized_path, expected)
        self.assertEqual(entries[0].encoding, "utf-16-be")


class SeratoWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.serato_root = self.base / "_Serato_"
        (self.serato_root / "SubCrates").mkdir(parents=True)
        self.audio_path = self.base / "test-song.mp3"
        self.audio_path.write_bytes(b"not-a-real-mp3-but-an-existing-test-file")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def crate_tracks(self, crate_name: str) -> list[str]:
        crate_file = self.serato_root / "SubCrates" / f"{crate_name}.crate"
        self.assertTrue(crate_file.is_file())
        return parse_serato_crate(str(crate_file))

    def test_dry_run_accepts_string_path_without_writing_files(self) -> None:
        results = write_tracks_to_crates(
            self.serato_root,
            [("Test Crate", str(self.audio_path))],
            dry_run=True,
            make_backup=True,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertFalse(results[0].changed)
        self.assertEqual(results[0].status, "dry_run")
        self.assertEqual(results[0].error, "")
        self.assertEqual(results[0].track_path, str(self.audio_path.resolve()))
        self.assertTrue((self.serato_root / "SubCrates").exists())
        self.assertFalse((self.base / "SeratoAI_Backups").exists())

    def test_dry_run_existing_track_is_successful_no_op(self) -> None:
        write_tracks_to_crates(self.serato_root, [("Test Crate", self.audio_path)], dry_run=False, make_backup=False)
        before = (self.serato_root / "SubCrates" / "Test Crate.crate").read_bytes()

        results = write_tracks_to_crates(
            self.serato_root, [("Test Crate", str(self.audio_path))], dry_run=True, make_backup=True
        )

        self.assertEqual([(r.success, r.changed, r.status) for r in results], [(True, False, "already_present")])
        self.assertEqual((self.serato_root / "SubCrates" / "Test Crate.crate").read_bytes(), before)
        self.assertFalse((self.base / "SeratoAI_Backups").exists())

    def test_live_write_adds_same_track_to_multiple_crates_without_database_changes(self) -> None:
        database = self.serato_root / "database V2"
        database.write_bytes(_crate_header())
        database_before = database.read_bytes()

        results = write_tracks_to_crates(
            self.serato_root,
            [("Test Root%%Crate A", self.audio_path), ("Test Crate B", str(self.audio_path))],
            dry_run=False,
            make_backup=True,
        )

        self.assertEqual(
            [(result.success, result.changed, result.status, result.error) for result in results],
            [(True, True, "added", ""), (True, True, "added", "")],
        )
        expected_path = str(self.audio_path.resolve())
        self.assertEqual(self.crate_tracks("Test Root%%Crate A"), [expected_path])
        self.assertEqual(self.crate_tracks("Test Crate B"), [expected_path])
        self.assertEqual(database.read_bytes(), database_before)
        backups = list((self.base / "SeratoAI_Backups").glob("_Serato__backup_*"))
        self.assertEqual(len(backups), 1)

    def test_temporary_root_integration_dry_run_then_live_write(self) -> None:
        """Exercise the exact controlled-test assignment shape end to end."""
        assignments = [
            ("Controlled One", str(self.audio_path)),
            ("Controlled One", self.audio_path),
            ("Controlled Two", str(self.audio_path)),
        ]
        before = sorted(
            (path.relative_to(self.serato_root), path.read_bytes())
            for path in self.serato_root.rglob("*")
            if path.is_file()
        )

        dry_results = write_tracks_to_crates(
            self.serato_root, assignments, dry_run=True, make_backup=True
        )

        self.assertEqual(
            [(result.success, result.changed, result.status) for result in dry_results],
            [(True, False, "dry_run"), (True, False, "dry_run")],
        )
        after_dry_run = sorted(
            (path.relative_to(self.serato_root), path.read_bytes())
            for path in self.serato_root.rglob("*")
            if path.is_file()
        )
        self.assertEqual(after_dry_run, before)
        self.assertFalse((self.base / "SeratoAI_Backups").exists())

        live_results = write_tracks_to_crates(
            self.serato_root, assignments, dry_run=False, make_backup=True
        )
        self.assertEqual(
            [(result.success, result.changed, result.status) for result in live_results],
            [(True, True, "added"), (True, True, "added")],
        )
        expected = [str(self.audio_path.resolve())]
        self.assertEqual(self.crate_tracks("Controlled One"), expected)
        self.assertEqual(self.crate_tracks("Controlled Two"), expected)
        first_entry = parse_serato_crate_entries(
            self.serato_root / "SubCrates" / "Controlled One.crate"
        )[0]
        self.assertEqual(first_entry.serialized_path, str(self.audio_path.resolve()).lstrip("/"))
        self.assertFalse((self.serato_root / "database V2").exists())
        self.assertEqual(len(list((self.base / "SeratoAI_Backups").glob("_Serato__backup_*"))), 1)

        repeated = write_tracks_to_crates(
            self.serato_root, assignments, dry_run=False, make_backup=True
        )
        self.assertEqual(
            [(result.success, result.changed, result.status) for result in repeated],
            [(True, False, "already_present"), (True, False, "already_present")],
        )
        self.assertEqual(self.crate_tracks("Controlled One"), expected)
        self.assertEqual(self.crate_tracks("Controlled Two"), expected)

    def test_existing_crate_track_is_not_written_twice(self) -> None:
        first = write_tracks_to_crates(
            self.serato_root, [("Test Crate", self.audio_path)], dry_run=False, make_backup=False
        )
        second = write_tracks_to_crates(
            self.serato_root, [("Test Crate", self.audio_path)], dry_run=False, make_backup=False
        )

        self.assertEqual((first[0].success, first[0].changed, first[0].status), (True, True, "added"))
        self.assertEqual((second[0].success, second[0].changed, second[0].status), (True, False, "already_present"))
        self.assertEqual(self.crate_tracks("Test Crate"), [str(self.audio_path.resolve())])

    def test_existing_serato_no_slash_path_matches_an_absolute_input(self) -> None:
        crate_dir = self.serato_root / "SubCrates"
        crate_file = crate_dir / "Existing.crate"
        crate_file.write_bytes(
            _crate_header() + serialize_serato_track_record(str(self.audio_path.resolve()).lstrip("/"))
        )
        before = crate_file.read_bytes()

        result = write_tracks_to_crates(
            self.serato_root, [("Existing", str(self.audio_path))], dry_run=False, make_backup=False
        )[0]

        self.assertEqual((result.success, result.changed, result.status), (True, False, "already_present"))
        self.assertEqual(crate_file.read_bytes(), before)
        self.assertEqual(self.crate_tracks("Existing"), [str(self.audio_path.resolve())])

    def test_duplicate_string_and_path_assignments_are_written_once(self) -> None:
        results = write_tracks_to_crates(
            self.serato_root,
            [("Test Crate", str(self.audio_path)), ("Test Crate", self.audio_path)],
            dry_run=False,
            make_backup=False,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].success, results[0].changed, results[0].status), (True, True, "added"))
        self.assertEqual(self.crate_tracks("Test Crate"), [str(self.audio_path.resolve())])

    def test_missing_audio_file_returns_a_failed_result(self) -> None:
        missing = self.base / "missing.mp3"
        results = write_tracks_to_crates(
            self.serato_root, [("Test Crate", missing)], dry_run=True, make_backup=False
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertFalse(results[0].changed)
        self.assertEqual(results[0].status, "failed")
        self.assertIn("Track file not found", results[0].error)
        self.assertEqual(results[0].track_path, str(missing.resolve()))

    def test_malformed_original_crate_is_not_replaced(self) -> None:
        crate_dir = self.serato_root / "SubCrates"
        crate_file = crate_dir / "Broken.crate"
        original = _crate_header() + _record(b"/")
        crate_file.write_bytes(original)

        results = write_tracks_to_crates(
            self.serato_root, [("Broken", self.audio_path)], dry_run=False, make_backup=True
        )

        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].success, results[0].changed, results[0].status), (False, False, "failed"))
        self.assertIn("expected an even UTF-16 byte length", results[0].error)
        self.assertEqual(crate_file.read_bytes(), original)
        self.assertFalse((self.base / "SeratoAI_Backups").exists())

    def test_unicode_normalization_has_one_comparison_key(self) -> None:
        nfc = str(self.base / "Beyoncé.mp3")
        nfd = unicodedata.normalize("NFD", nfc)
        _, nfc_key = canonical_track_path(nfc)
        _, nfd_key = canonical_track_path(nfd)
        self.assertEqual(nfc_key, nfd_key)

    def test_read_only_diagnostic_reports_crate_references_and_database_state(self) -> None:
        write_tracks_to_crates(
            self.serato_root,
            [("First", self.audio_path), ("Second", self.audio_path)],
            dry_run=False,
            make_backup=False,
        )

        report = diagnose_serato_duplicates(self.serato_root)
        self.assertEqual(report["duplicate_entries_in_one_crate"], [])
        self.assertEqual(len(report["references_in_multiple_crates"]), 1)
        self.assertFalse(report["database_v2"]["exists"])
        raw_paths = [record["serialized_path"] for record in report["crate_records"]]
        self.assertEqual(
            raw_paths,
            [str(self.audio_path.resolve()).lstrip("/"), str(self.audio_path.resolve()).lstrip("/")],
        )


if __name__ == "__main__":
    unittest.main()
