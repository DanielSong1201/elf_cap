from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load_script("build_caption_overfit_subset")
ANALYZE = load_script("analyze_caption_overfit")
DIAGNOSTICS = load_script("caption_diagnostics_common")
FLOW_RECOVERY = load_script("diagnose_flow_recovery")


def passthrough_tqdm(iterable, **_kwargs):
    return iterable


class CaptionOverfitExperimentTest(unittest.TestCase):
    def test_flow_recovery_times_are_validated_and_sorted(self) -> None:
        self.assertEqual(
            FLOW_RECOVERY.validate_start_times([0.9, 0.0, 0.5, 0.5]),
            [0.0, 0.5, 0.9],
        )
        with self.assertRaises(ValueError):
            FLOW_RECOVERY.validate_start_times([-0.1, 0.5])

    def test_flow_recovery_uniform_steps_have_exact_endpoints(self) -> None:
        import torch

        steps = FLOW_RECOVERY.build_recovery_steps(
            0.25, 4, "uniform", None, torch.device("cpu"), torch.float32
        )
        self.assertEqual(len(steps), 5)
        self.assertAlmostEqual(float(steps[0]), 0.25)
        self.assertAlmostEqual(float(steps[-1]), 1.0)
        self.assertTrue(bool((steps[1:] > steps[:-1]).all()))

    def test_flow_recovery_latent_metrics_are_exact_for_clean_input(self) -> None:
        import torch

        target = torch.randn(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
        metrics = FLOW_RECOVERY.masked_latent_metrics(target, target, mask)
        self.assertEqual(metrics["mse"], [0.0, 0.0])
        self.assertEqual(metrics["relative_l2_error"], [0.0, 0.0])
        for cosine in metrics["cosine_similarity"]:
            self.assertAlmostEqual(cosine, 1.0, places=6)

    def test_diagnostic_generation_metrics_detect_exact_and_repeated_text(self) -> None:
        result = DIAGNOSTICS.generation_metrics(
            ["A dog barks.", "cat cat cat cat", ""],
            ["A dog barks.", "A cat meows."],
        )
        self.assertAlmostEqual(result["nonempty_rate"], 2 / 3)
        self.assertEqual(result["exact_match_rate"], 0.5)
        self.assertEqual(result["training_caption_coverage"], 0.5)
        self.assertEqual(result["repeated_bigram_rate"], 0.5)

    def test_first_eos_positions_marks_missing_eos(self) -> None:
        import torch

        ids = torch.tensor([[4, 1, 1], [3, 2, 1], [9, 8, 7]])
        positions = DIAGNOSTICS.first_eos_positions(ids, eos_id=1)
        self.assertEqual(positions.tolist(), [1, 2, -1])

    def test_diagnostic_batch_conversion_accepts_numpy_arrays(self) -> None:
        import numpy as np
        import torch

        value = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = DIAGNOSTICS.batch_value_to_tensor(
            value, torch.device("cpu"), dtype=torch.long
        )
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(tensor.tolist(), value.tolist())

    def test_diagnostic_launcher_calls_three_independent_scripts(self) -> None:
        launcher = (REPO_ROOT / "scripts" / "run_caption_diagnostics.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("diagnose_latent_reconstruction.py", launcher)
        self.assertIn("diagnose_raw_vs_ema.py", launcher)
        self.assertIn("diagnose_sampler_sweep.py", launcher)
        self.assertNotIn("scripts/run_caption_diagnostics.py", launcher)

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

    def test_launcher_keeps_path_overrides_as_single_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"build_caption_overfit_subset.py"* ]]; then
    output=""
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--output-dir" ]]; then
            output=$2
            break
        fi
        shift
    done
    mkdir -p "${output}/train"
    : > "${output}/train/dataset_info.json"
    : > "${output}/references.jsonl"
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            experiment_root = root / "experiment with spaces"
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "EXPERIMENT_ROOT": str(experiment_root),
                    "SOURCE_DATA": str(root / "source data"),
                    "EMA_DECAY": "0.99",
                }
            )
            result = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "run_caption_overfit_100.sh")],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("No such file or directory", result.stderr)
            self.assertIn(str(experiment_root / "data" / "train"), result.stdout)


if __name__ == "__main__":
    unittest.main()
