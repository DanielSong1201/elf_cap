from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_elf_caption_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_elf_caption_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

try:
    import datasets
except ImportError:
    datasets = None


def passthrough_tqdm(iterable, **_kwargs):
    return iterable


class FakeTokenizer:
    eos_token_id = 1

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=48,
        padding=False,
    ):
        del truncation, padding
        encoded = []
        for text in texts:
            ids = [3 + (ord(character) % 50) for character in text]
            if add_special_tokens:
                ids = ids[: max_length - 1] + [self.eos_token_id]
            else:
                ids = ids[:max_length]
            encoded.append(ids)
        return {"input_ids": encoded}


class PrepareElfCaptionDataTest(unittest.TestCase):
    def test_expand_normalize_and_skip_invalid_captions(self) -> None:
        summary = {"skipped": {}, "skipped_examples": {}}
        record = {
            "youtube_id": "abc",
            "audio_id": "Yabc",
            "audio_path": "data/AudioCaps_CVSSP/val/Yabc.wav",
            "start_time": 10,
            "source_split": "val",
            "captions": [
                "  A dog   barks.  ",
                "A dog barks.",
                "Birds sing.\n",
                "   ",
                None,
            ],
            "audiocap_ids": ["1", "2", "3", "4", "5"],
        }
        examples = MODULE.validate_manifest_record(
            record, "eval", 1, summary, max_examples=10
        )
        self.assertEqual([item["caption"] for item in examples], ["A dog barks.", "Birds sing."])
        self.assertEqual(examples[0]["audiocap_id"], "1")
        self.assertEqual(examples[1]["caption_index"], 2)
        self.assertEqual(summary["skipped"]["duplicate_caption_within_audio"], 1)
        self.assertEqual(summary["skipped"]["empty_caption"], 1)
        self.assertEqual(summary["skipped"]["non_string_caption"], 1)

    def test_load_manifest_assigns_stable_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "train.jsonl"
            rows = [
                {"youtube_id": "a", "captions": ["One.", "Two."]},
                {"youtube_id": "b", "captions": []},
                {"youtube_id": "c", "captions": ["Three."]},
            ]
            manifest.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            examples, summary = MODULE.load_and_expand_manifest(
                manifest, "train", 10, passthrough_tqdm
            )
            self.assertEqual([item["index"] for item in examples], [0, 1, 2])
            self.assertEqual(summary["manifest_records"], 3)
            self.assertEqual(summary["caption_examples"], 3)
            self.assertEqual(summary["unique_audio"], 2)
            self.assertEqual(summary["skipped"]["missing_caption_list"], 1)

    def test_default_worker_count_is_bounded(self) -> None:
        self.assertGreaterEqual(MODULE.default_num_workers(), 1)
        self.assertLessEqual(MODULE.default_num_workers(), 16)

    @unittest.skipIf(datasets is None, "datasets is not installed in this environment")
    def test_tokenize_and_save_arrow_dataset_with_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            examples = [
                {
                    "index": index,
                    "caption": f"Caption number {index}.",
                    "youtube_id": f"id{index}",
                    "audio_id": f"Yid{index}",
                    "audio_path": f"data/Yid{index}.wav",
                    "start_time": 0,
                    "split": "train",
                    "source_split": "train",
                    "caption_index": 0,
                    "audiocap_id": str(index),
                }
                for index in range(8)
            ]
            summary = MODULE.tokenize_and_save(
                "train",
                examples,
                output_dir,
                FakeTokenizer(),
                max_length=12,
                num_workers=2,
                batch_size=4,
            )
            loaded = datasets.load_from_disk(summary["hf_dataset"])
            self.assertEqual(len(loaded), 8)
            self.assertIn("input_ids", loaded.column_names)
            self.assertIn("sequence_length", loaded.column_names)
            self.assertTrue(all(length <= 12 for length in loaded["sequence_length"]))


if __name__ == "__main__":
    unittest.main()
