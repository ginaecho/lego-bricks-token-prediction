import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import implementation as impl


ROOT = Path(__file__).resolve().parent


def payload(*texts):
    return {"schema_version": 1, "dataset_label": "synthetic",
            "feedback": [{"id": f"synthetic-{i}", "text": text}
                         for i, text in enumerate(texts)]}


class FeedbackTests(unittest.TestCase):
    def cli(self, *args):
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *map(str, args)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        return completed.returncode, json.loads(completed.stdout)

    def test_example(self):
        data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))
        result = impl.analyze(data)
        self.assertEqual(result["summary"], {
            "input_count": 5, "unique_count": 4, "duplicate_count": 1})
        self.assertEqual(len(result["themes"]), 6)
        self.assertEqual(result["themes"][0]["original_count"], 2)

    def test_normalized_duplicates_and_traceability(self):
        data = payload("Ｑｕａｌｉｔｙ, GREAT!", "quality great", "QUALITY   great.")
        result = impl.analyze(data)
        self.assertEqual(result["summary"]["unique_count"], 1)
        self.assertEqual(result["groups"][0]["member_ids"],
                         ["synthetic-0", "synthetic-1", "synthetic-2"])
        item = result["themes"][0]["evidence"][0]
        self.assertEqual(data["feedback"][0]["text"][item["start"]:item["end"]],
                         item["excerpt"])

    def test_empty(self):
        result = impl.analyze(payload())
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["themes"], [])
        self.assertEqual(result["summary"]["input_count"], 0)

    def test_multiple_themes_and_fallback(self):
        result = impl.analyze(payload("Delivery is expensive.", "Purple packaging!",
                                      "Supporting craftsmanship"))
        self.assertEqual(result["groups"][0]["themes"], ["delivery", "price"])
        self.assertEqual(result["groups"][1]["themes"], ["other"])
        self.assertEqual(result["groups"][2]["themes"], ["other"])

    def test_semantic_variants_remain_distinct(self):
        result = impl.analyze(payload("Shipping is late", "Delivery was delayed"))
        self.assertEqual(result["summary"]["unique_count"], 2)

    def test_deterministic_and_does_not_mutate(self):
        data = payload("easy and affordable", "durable quality")
        before = copy.deepcopy(data)
        self.assertEqual(impl.analyze(data), impl.analyze(data))
        self.assertEqual(data, before)

    def test_invalid_inputs(self):
        invalid = [None, [], {}, payload(""), payload("!!!"), payload("   "),
                   payload("a" * 4001), payload(12)]
        for key, value in [("schema_version", True), ("schema_version", 2),
                           ("feedback", {}), ("dataset_label", "real"), ("extra", 1)]:
            data = payload("quality")
            data[key] = value
            invalid.append(data)
        for identifier in ["", "  ", " untrimmed", 3]:
            data = payload("quality")
            data["feedback"][0]["id"] = identifier
            invalid.append(data)
        duplicate = payload("quality", "support")
        duplicate["feedback"][1]["id"] = duplicate["feedback"][0]["id"]
        invalid.append(duplicate)
        for data in invalid:
            with self.subTest(data=str(data)[:100]):
                with self.assertRaises(impl.ValidationError):
                    impl.analyze(data)

    def test_limits(self):
        self.assertEqual(impl.analyze(payload("a" * 4000))["status"], "ok")
        data = payload("valid")
        data["feedback"] *= 10001
        with self.assertRaises(impl.ValidationError):
            impl.analyze(data)

    def test_output_validation_rejects_fabricated_excerpt(self):
        data = payload("helpful support")
        result = impl.analyze(data)
        result["themes"][0]["evidence"][0]["excerpt"] = "fabricated"
        with self.assertRaises(impl.ValidationError):
            impl.validate(result, "output", data)

    def test_cli_success(self):
        code, result = self.cli(ROOT / "example_input.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")

    def test_cli_file_usage_and_parse_errors(self):
        for args in [(), (ROOT / "nonexistent-input.json",),
                     (ROOT / "implementation.py",), (ROOT,), ("a", "b")]:
            with self.subTest(args=args):
                code, result = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "error")

    def test_json_and_encoding_validation(self):
        for text in ['{"schema_version":1,"schema_version":1}',
                     '{"schema_version":NaN}', '{"feedback":', '[]']:
            with self.subTest(text=text):
                with patch.object(impl.Path, "read_text", return_value=text):
                    with patch("builtins.print") as output:
                        self.assertEqual(impl.main(["synthetic.json"]), 2)
                        self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")
        with patch.object(impl.Path, "read_text",
                          side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            with patch("builtins.print") as output:
                self.assertEqual(impl.main(["synthetic.json"]), 2)
                self.assertEqual(json.loads(output.call_args.args[0])["status"], "error")


if __name__ == "__main__":
    unittest.main()
