from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load_script("build_caption_overfit_subset")
ANALYZE = load_script("analyze_caption_overfit")


def passthrough_tqdm(iterable, **_kwargs):
    return iterable


class CaptionOverfitExperimentTest(unittest.TestCase):
    def test_subset_selection_is_deterministic_and_sorted(self) -> None:
        first = BUILD.select_indices(1000, 100, 42)
        second = BUILD.select_indices(1000, 100, 42)
        different = BUILD.select_indices(1000, 100, 43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), len(set(first)))

    def test_subset_selection_rejects_insufficient_data(self) -> None:
        with self.assertRaises(ValueError):
            BUILD.select_indices(99, 100, 42)

    def test_generation_analysis_detects_memorization_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated_path = Path(temporary) / "all_generated_20_200.jsonl"
            generations = [
                {"id": 0, "generated": "A dog barks."},
                {"id": 1, "generated": "Birds are singing loudly."},
                {"id": 2, "generated": "dog dog dog dog"},
                {"id": 3, "generated": ""},
            ]
            generated_path.write_text(
                "\n".join(json.dumps(row) for row in generations) + "\n",
                encoding="utf-8",
            )
            result = ANALYZE.analyze_generation(
                generated_path,
                ["A dog barks.", "Birds sing."],
                max_examples=5,
                tqdm_class=passthrough_tqdm,
            )
            self.assertEqual(result["epoch"], 20)
            self.assertEqual(result["step"], 200)
            self.assertEqual(result["num_nonempty"], 3)
            self.assertAlmostEqual(result["exact_match_rate"], 1 / 3)
            self.assertEqual(result["training_caption_coverage"], 0.5)
            self.assertGreater(result["repeated_bigram_rate"], 0)

    def test_training_log_parser_reads_tqdm_log_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                "INFO - engine - Step 10: loss=2.5000, l2=1.2000, ce=7.7000, lr=1e-4\n"
                "INFO - engine - Step 20: loss=1.2500, l2=0.6000, ce=4.0000, lr=1e-4\n",
                encoding="utf-8",
            )
            result = ANALYZE.parse_training_log(log_path, passthrough_tqdm)
            assert result is not None
            self.assertEqual(result["points"], 2)
            self.assertEqual(result["last"]["step"], 20)
            self.assertEqual(result["minimum_loss"]["loss"], 1.25)


if __name__ == "__main__":
    unittest.main()
