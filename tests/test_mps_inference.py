"""Apple GPU inference must match CPU answers, and memory admission must follow the device."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mdtrain.engine import DecisionModel, admit_host_memory
from mdtrain.fixtures import make_fixtures
from mdtrain.preflight import smoke_tiny
from mdtrain.train import deterministic

ROOT = Path(__file__).resolve().parents[1]
BASE_CACHE = ROOT / "data/raw/processor-cache/models--Qwen--Qwen3-VL-4B-Instruct"
BASE_WEIGHT_BYTES = int(8.27 * 1024 ** 3)
GIB = 1024 ** 3


@unittest.skipUnless(torch.backends.mps.is_available(), "needs an Apple Silicon GPU")
@unittest.skipUnless(BASE_CACHE.is_dir(), "needs the downloaded Qwen3-VL-4B-Instruct processor files")
class MpsInferenceTests(unittest.TestCase):
    def test_mps_answers_match_cpu_on_text_image_and_video(self):
        deterministic()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = Path(smoke_tiny(ROOT, directory / "tiny.json")["artifact"])
            requests = [example.input for example in make_fixtures(directory / "media")]
            answers = {device: DecisionModel.load(artifact, device=device, root=ROOT,
                                                  media_root=directory / "media").predict(requests)
                       for device in ("cpu", "mps")}
        self.assertEqual([row["id"] for row in answers["mps"]],
                         ["fixture-text", "fixture-image", "fixture-mixed", "fixture-video"])
        for cpu, mps in zip(answers["cpu"], answers["mps"]):
            self.assertEqual(mps["predicted_key"], cpu["predicted_key"])
            for key, probability in cpu["probabilities"].items():
                self.assertAlmostEqual(mps["probabilities"][key], probability, delta=1e-4)


class MemoryAdmissionTests(unittest.TestCase):
    def admit(self, device, *, total, available):
        memory = SimpleNamespace(total=total * GIB, available=available * GIB)
        warnings = io.StringIO()
        with mock.patch("mdtrain.engine.psutil.virtual_memory", return_value=memory), \
                contextlib.redirect_stderr(warnings):
            admit_host_memory(BASE_WEIGHT_BYTES, device)
        return warnings.getvalue()

    def test_mps_refuses_a_mac_whose_total_memory_is_too_small(self):
        with self.assertRaisesRegex(MemoryError, r"on mps needs 11\.8 GiB of unified memory; this Mac has 8\.0 GiB"):
            self.admit("mps", total=8, available=6)

    def test_mps_loads_with_a_swap_warning_when_free_memory_is_short(self):
        warning = self.admit("mps", total=16, available=3)
        self.assertIn("3.0 GiB is free and the model is best with 11.8 GiB, so macOS will swap", warning)

    def test_mps_loads_silently_when_free_memory_is_enough(self):
        self.assertEqual(self.admit("mps", total=16, available=12), "")

    def test_cpu_still_refuses_short_free_memory(self):
        with self.assertRaisesRegex(MemoryError, r"on cpu needs 11\.8 GiB of free memory; 3\.0 GiB is available"):
            self.admit("cpu", total=16, available=3)


if __name__ == "__main__":
    unittest.main()
