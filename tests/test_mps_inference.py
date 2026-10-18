"""Apple GPU inference must match CPU answers and refuse loads that would not fit in memory."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mdtrain.engine import DecisionModel, load_body
from mdtrain.fixtures import make_fixtures
from mdtrain.preflight import smoke_tiny
from mdtrain.train import deterministic

ROOT = Path(__file__).resolve().parents[1]
BASE_CACHE = ROOT / "data/raw/processor-cache/models--Qwen--Qwen3-VL-4B-Instruct"


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

    def test_mps_refuses_a_load_that_does_not_fit_free_memory(self):
        three_gib = SimpleNamespace(available=3 * 1024 ** 3)
        with mock.patch("mdtrain.engine.psutil.virtual_memory", return_value=three_gib):
            with self.assertRaisesRegex(MemoryError, r"on mps needs 11\.8 GiB of free memory; 3\.0 GiB"):
                load_body(ROOT, device="mps")


if __name__ == "__main__":
    unittest.main()
