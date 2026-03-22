import json
import subprocess
import tempfile
import unittest
from pathlib import Path


TRAIN_SCRIPT = Path("scripts/train_tokenizer.py")
RESOLVE_SCRIPT = Path("scripts/resolve_tokenizer.py")
TRAIN_FIXTURE = Path("tests/fixtures/tokenizer_train_small.txt")
REQUIRED_SPECIAL_TOKENS = [
    "<gap>",
    "[LETTER]",
    "[DEBT_NOTE]",
    "[CONTRACT]",
    "[ADMIN]",
]


class TestTokenizerTrainingContract(unittest.TestCase):
    def test_training_writes_artifacts_and_manifest_has_special_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = subprocess.run(
                [
                    "python",
                    str(TRAIN_SCRIPT),
                    "--input-file",
                    str(TRAIN_FIXTURE),
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            combined_output = f"{result.stdout}\n{result.stderr}"
            self.assertEqual(0, result.returncode, msg=combined_output)

            model_path = output_dir / "akkadian_sp.model"
            vocab_path = output_dir / "akkadian_sp.vocab"
            manifest_path = output_dir / "tokenizer_manifest.json"
            self.assertTrue(model_path.exists(), msg="missing akkadian_sp.model")
            self.assertTrue(vocab_path.exists(), msg="missing akkadian_sp.vocab")
            self.assertTrue(
                manifest_path.exists(), msg="missing tokenizer_manifest.json"
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(REQUIRED_SPECIAL_TOKENS, manifest.get("special_tokens"))

    def test_resolver_emits_required_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            train_result = subprocess.run(
                [
                    "python",
                    str(TRAIN_SCRIPT),
                    "--input-file",
                    str(TRAIN_FIXTURE),
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            train_output = f"{train_result.stdout}\n{train_result.stderr}"
            self.assertEqual(0, train_result.returncode, msg=train_output)

            manifest_path = output_dir / "tokenizer_manifest.json"
            resolve_result = subprocess.run(
                [
                    "python",
                    str(RESOLVE_SCRIPT),
                    "--manifest",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            resolve_output = f"{resolve_result.stdout}\n{resolve_result.stderr}"
            self.assertEqual(0, resolve_result.returncode, msg=resolve_output)

            payload = json.loads(resolve_result.stdout)
            self.assertIn("model_path", payload)
            self.assertIn("vocab_path", payload)
            self.assertIn("special_tokens", payload)
            self.assertEqual(REQUIRED_SPECIAL_TOKENS, payload["special_tokens"])


if __name__ == "__main__":
    unittest.main()
