import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch

from implementation import InputError, analyze, main, unique_object, reject_constant


def payload(text="good", **extras):
    return {"feedback": [{"id": "original-1", "text": text, **extras}]}


class SentimentTests(unittest.TestCase):
    def test_labeled_fixtures(self):
        fixtures = [
            ("good", "positive", 0.7), ("bad", "negative", -0.7),
            ("not good", "negative", -0.7), ("not bad", "positive", 0.7),
            ("not never good", "positive", 0.7), ("quux blue", "neutral", 0),
            ("good bad", "neutral", 0), ("GOOD", "positive", 0.7),
            ("isn’t good", "negative", -0.7),
        ]
        for text, label, score in fixtures:
            with self.subTest(text=text):
                sentiment = analyze(payload(text))["feedback"][0]["sentiment"]
                self.assertEqual(sentiment["label"], label)
                self.assertAlmostEqual(sentiment["score"], score)

    def test_negation_window_and_boundary(self):
        for text in ["not. good", "not! good", "not; good", "not\n good",
                     "not a b c good"]:
            with self.subTest(text=text):
                self.assertEqual(analyze(payload(text))["feedback"][0]["sentiment"]["score"], 0.7)
        self.assertEqual(analyze(payload("not a b good"))["feedback"][0]["sentiment"]["score"], -0.7)

    def test_config_overrides(self):
        data = payload("good splendid not splendid")
        data["config"] = {"lexicon": {"splendid": 0.8}, "negation_window": 0}
        result = analyze(data)["feedback"][0]["sentiment"]
        self.assertEqual(result["matched_terms"], 2)
        self.assertEqual(result["score"], 0.8)

    def test_empty(self):
        self.assertEqual(analyze({"feedback": []})["status"], "empty")
        row = analyze(payload(" \n "))["feedback"][0]
        self.assertEqual(row["status"], "empty_text")
        self.assertEqual(row["sentiment"]["label"], "neutral")

    def test_source_offsets(self):
        text = "Hello!  This is not good. Next."
        evidence = analyze(payload(text))["feedback"][0]["sentiment"]["evidence"][0]
        self.assertEqual(text[evidence["start"]:evidence["end"]], "good")
        source = evidence["excerpt"]
        self.assertEqual(source["text"], text[source["start"]:source["end"]])
        self.assertEqual(source["text"], "  This is not good.")
        self.assertTrue(evidence["negated"])

    def test_frequency_and_priority(self):
        data = {"feedback": [
            {"id": "a", "text": "slow slow", "severity": 3, "urgency": 5},
            {"id": "b", "text": "slow", "severity": 1, "urgency": 1},
        ]}
        issue = analyze(data)["prioritized_issues"][0]
        self.assertEqual(issue["frequency"], 2)
        self.assertEqual(issue["mean_severity"], 2)
        self.assertEqual(issue["mean_urgency"], 3)
        self.assertAlmostEqual(issue["priority_score"], 16)
        self.assertEqual(len(issue["sources"][0]["matches"]), 2)

    def test_ties(self):
        data = payload("x")
        data["config"] = {"issues": {"z": ["x"], "a": ["x"]}}
        self.assertEqual([row["issue"] for row in analyze(data)["prioritized_issues"]], ["a", "z"])
        self.assertEqual(analyze(data), analyze(data))

    def test_multiword_and_whole_word(self):
        data = payload("slowly; LOAD time. load. time")
        data["config"] = {"issues": {"latency": ["load time"], "speed": ["slow"]}}
        issues = analyze(data)["prioritized_issues"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(len(issues[0]["sources"][0]["matches"]), 1)
        self.assertEqual(issues[0]["sources"][0]["matches"][0]["excerpt"]["text"], " LOAD time.")

    def test_no_issues_and_empty_config(self):
        data = payload("slow")
        data["config"] = {"lexicon": {}, "issues": {}}
        result = analyze(data)
        self.assertEqual(result["prioritized_issues"], [])
        self.assertEqual(result["feedback"][0]["sentiment"]["score"], 0)

    def test_malformed_inputs(self):
        invalid = [None, [], {}, {"feedback": {}}, {"feedback": [None]},
                   {"feedback": [{"id": "x"}]}, {"feedback": [{"id": [], "text": "x"}]},
                   {"feedback": [{"id": "", "text": "x"}]},
                   payload(text=None), payload(severity=True), payload(urgency=6),
                   payload(severity=float("nan")), payload(urgency=-1),
                   payload(extra=1), {"feedback": [], "extra": 1}]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(InputError):
                analyze(data)

    def test_duplicate_ids(self):
        data = payload()
        data["feedback"] *= 2
        with self.assertRaises(InputError):
            analyze(data)

    def test_huge_number_rejected(self):
        with self.assertRaises(InputError):
            analyze(payload(severity=10 ** 1000))

    def test_malformed_config(self):
        invalid = [
            None, {"unknown": 1}, {"lexicon": []}, {"lexicon": {"a": 2}},
            {"lexicon": {"a": True}}, {"lexicon": {"A": 1, "a": 0}},
            {"lexicon": {"two words": 1}}, {"issues": []},
            {"issues": {"": ["x"]}}, {"issues": {"a": []}},
            {"issues": {"a": [""]}}, {"issues": {"a": ["load.time"]}},
            {"negations": "no"}, {"negations": [7]},
            {"negation_window": True}, {"negation_window": -1},
        ]
        for config in invalid:
            data = payload()
            data["config"] = config
            with self.subTest(config=config), self.assertRaises(InputError):
                analyze(data)

    def test_classifier_labeled_fixtures(self):
        fixtures = {"p": ("good", "positive"), "n": ("not good", "negative"),
                    "u": ("blue", "neutral")}
        data = {"feedback": [{"id": key, "text": text} for key, (text, _) in fixtures.items()]}
        def callback(rows):
            self.assertEqual(rows, data["feedback"])
            return [{"id": row["id"], "label": fixtures[row["id"]][1], "confidence": 0.9}
                    for row in reversed(rows)]
        result = analyze(data, callback)
        for row in result["feedback"]:
            self.assertEqual(row["classifier"]["label"], fixtures[row["id"]][1])
            self.assertEqual(row["sentiment"]["label"], row["classifier"]["label"])

    def test_classifier_annotations_do_not_override(self):
        result = analyze(payload("slow"), lambda _: [
            {"id": "original-1", "label": "positive", "confidence": 1}])
        self.assertEqual(result["feedback"][0]["sentiment"]["label"], "negative")
        self.assertEqual(result["prioritized_issues"], analyze(payload("slow"))["prioritized_issues"])

    def test_classifier_validation(self):
        valid = {"id": "original-1", "label": "positive", "confidence": 0.9}
        invalid = [None, {}, [], [valid, valid], [{**valid, "id": "invented"}],
                   [{**valid, "label": "happy"}], [{**valid, "label": []}],
                   [{**valid, "confidence": True}], [{**valid, "confidence": -0.1}],
                   [{**valid, "confidence": 1.1}], [{**valid, "confidence": float("inf")}],
                   [{"id": "original-1", "label": "neutral"}], [{**valid, "extra": 1}]]
        for results in invalid:
            with self.subTest(results=results), self.assertRaises(InputError):
                analyze(payload(), lambda _: results)

    def test_classifier_exception(self):
        def callback(_):
            raise RuntimeError("failure")
        with self.assertRaisesRegex(InputError, "callback failed"):
            analyze(payload(), callback)

    def test_empty_classifier_batch(self):
        self.assertEqual(analyze({"feedback": []}, lambda rows: rows)["status"], "empty")

    def test_classifier_cannot_mutate_originals(self):
        data = payload()
        def callback(rows):
            rows[0]["text"] = "bad"
            return [{"id": rows[0]["id"], "label": "negative", "confidence": 1}]
        self.assertEqual(analyze(data, callback)["feedback"][0]["sentiment"]["label"], "positive")
        self.assertEqual(data["feedback"][0]["text"], "good")

    def test_strict_json(self):
        for raw in ['{"feedback":[],"feedback":[]}', '{"feedback":NaN}', '{"feedback":']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)

    def test_cli_example(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(Path(__file__).with_name("example_input.json"))])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["feedback_count"], 4)
        self.assertEqual(result["prioritized_issues"][0]["issue"], "reliability")

    def test_cli_missing_file(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main([str(Path(__file__).with_name("nonexistent-input.json"))])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_cli_malformed_file(self):
        for raw in ["", "{", '{"feedback":[],"feedback":[]}', '{"feedback":NaN}',
                    '{"feedback":[{"id":"x","text":1}]}']:
            stderr, stdout = io.StringIO(), io.StringIO()
            with self.subTest(raw=raw), patch("builtins.open", mock_open(read_data=raw)):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["fixture.json"]), 2)
                self.assertEqual(json.loads(stderr.getvalue())["status"], "error")
                self.assertEqual(stdout.getvalue(), "")

    def test_cli_output_file(self):
        mocked = mock_open(read_data='{"feedback":[]}')
        with patch("builtins.open", mocked):
            self.assertEqual(main(["fixture.json", "--output", "result.json"]), 0)
        result = json.loads(mocked().write.call_args.args[0])
        self.assertEqual(result["status"], "empty")


if __name__ == "__main__":
    unittest.main()
