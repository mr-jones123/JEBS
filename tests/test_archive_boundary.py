import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from mdprep.package import _extract


class ArchiveBoundaryTests(unittest.TestCase):
    def test_regular_member_is_verified_and_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bundle.tar.gz"
            payload = b"verified archive contents"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("data/example.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            expected = {"data/example.txt": {"bytes": len(payload),
                                             "sha256": hashlib.sha256(payload).hexdigest()}}
            _extract(archive_path, root / "extracted", expected)
            self.assertEqual((root / "extracted/data/example.txt").read_bytes(), b"verified archive contents")

    def test_parent_traversal_cannot_write_outside_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bundle.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("../escaped.txt")
                member.size = 3
                archive.addfile(member, io.BytesIO(b"bad"))
            expected = {"../escaped.txt": {"bytes": 3, "sha256": hashlib.sha256(b"bad").hexdigest()}}
            with self.assertRaises((ValueError, tarfile.FilterError)):
                _extract(archive_path, root / "extracted", expected)
            self.assertFalse((root / "escaped.txt").exists())

    def test_link_member_is_not_materialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bundle.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("data/link")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            expected = {"data/link": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}
            with self.assertRaises((ValueError, tarfile.FilterError)):
                _extract(archive_path, root / "extracted", expected)
            self.assertFalse((root / "extracted/data/link").is_symlink())


if __name__ == "__main__":
    unittest.main()
