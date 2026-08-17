from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_audiocaps_manifests.py"


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 160)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["audiocap_id", "youtube_id", "start_time", "caption"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class BuildAudioCapsManifestsTest(unittest.TestCase):
    def test_official_metadata_layout_aggregates_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root = root / "audio"
            metadata_root = root / "metadata"
            output_dir = root / "manifests"

            for split in ("train", "val", "test"):
                (audio_root / split).mkdir(parents=True)

            write_wav(audio_root / "train" / "Ygood.wav")
            write_wav(audio_root / "train" / "Yno_caption.wav")
            write_csv(
                metadata_root / "train.csv",
                [
                    {
                        "audiocap_id": "1",
                        "youtube_id": "good",
                        "start_time": "10",
                        "caption": "A bell rings.",
                    },
                    {
                        "audiocap_id": "2",
                        "youtube_id": "missing",
                        "start_time": "20",
                        "caption": "A dog barks.",
                    },
                    {
                        "audiocap_id": "3",
                        "youtube_id": "no_caption",
                        "start_time": "30",
                        "caption": "  ",
                    },
                ],
            )

            write_wav(audio_root / "val" / "Yshared.wav")
            write_csv(
                metadata_root / "val.csv",
                [
                    {
                        "audiocap_id": "4",
                        "youtube_id": "shared",
                        "start_time": "0",
                        "caption": "Rain falls.",
                    },
                    {
                        "audiocap_id": "5",
                        "youtube_id": "shared",
                        "start_time": "0",
                        "caption": "Water patters on a surface.",
                    },
                    {
                        "audiocap_id": "6",
                        "youtube_id": "shared",
                        "start_time": "0",
                        "caption": "Rain falls.",
                    },
                ],
            )

            (audio_root / "test" / "Ybroken.wav").write_bytes(b"not a wav")
            write_csv(
                metadata_root / "test.csv",
                [
                    {
                        "audiocap_id": "7",
                        "youtube_id": "broken",
                        "start_time": "5",
                        "caption": "Noise is audible.",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--audio-root",
                    str(audio_root),
                    "--metadata-root",
                    str(metadata_root),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            train = read_jsonl(output_dir / "train.jsonl")
            evaluation = read_jsonl(output_dir / "eval.jsonl")
            test = read_jsonl(output_dir / "test.jsonl")
            self.assertEqual(len(train), 1)
            self.assertEqual(train[0]["youtube_id"], "good")
            self.assertEqual(train[0]["start_time"], 10)
            self.assertEqual(train[0]["captions"], ["A bell rings."])
            self.assertEqual(evaluation[0]["split"], "eval")
            self.assertEqual(evaluation[0]["source_split"], "val")
            self.assertEqual(
                evaluation[0]["captions"],
                ["Rain falls.", "Water patters on a surface."],
            )
            self.assertEqual(test, [])

            summary = json.loads((output_dir / "summary.json").read_text())
            summaries = {item["manifest_split"]: item for item in summary["splits"]}
            self.assertEqual(summaries["train"]["skipped"]["missing_audio"], 1)
            self.assertEqual(summaries["train"]["skipped"]["empty_caption_rows"], 1)
            self.assertEqual(
                summaries["train"]["skipped"]["audio_without_valid_caption"], 1
            )
            self.assertEqual(summaries["eval"]["skipped"]["duplicate_caption_rows"], 1)
            self.assertEqual(summaries["test"]["skipped"]["invalid_wav_header"], 1)

    def test_split_local_csv_and_ambiguous_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "AudioCaps_CVSSP"
            output_dir = root / "manifests"

            for split in ("train", "val", "test"):
                write_wav(data_root / split / "Ysame.wav")
                rows = [
                    {
                        "audiocap_id": "1",
                        "youtube_id": "same",
                        "start_time": "0",
                        "caption": "First caption.",
                    }
                ]
                if split == "train":
                    rows.append(
                        {
                            "audiocap_id": "2",
                            "youtube_id": "same",
                            "start_time": "10",
                            "caption": "A different clip.",
                        }
                    )
                write_csv(data_root / split / "captions.csv", rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--audio-root",
                    str(data_root),
                    "--output-dir",
                    str(output_dir),
                    "--absolute-audio-paths",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_jsonl(output_dir / "train.jsonl"), [])
            self.assertTrue(Path(read_jsonl(output_dir / "eval.jsonl")[0]["audio_path"]).is_absolute())
            summary = json.loads((output_dir / "summary.json").read_text())
            train_summary = next(
                item for item in summary["splits"] if item["manifest_split"] == "train"
            )
            self.assertEqual(train_summary["skipped"]["ambiguous_start_time_audio"], 1)


if __name__ == "__main__":
    unittest.main()
