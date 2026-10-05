import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import implementation as app


ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example_input.json").read_text(encoding="utf-8"))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "implementation.py"), *map(str, args)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def test_grounded_support(self):
        result = app.run(self.data)
        self.assertEqual(result["support"]["citations"], ["synthetic-return"])
        self.assertEqual(result["support"]["answer"], self.data["knowledge"][0]["answer"])
        self.assertFalse(result["support"]["needs_human"])

    def test_synonym_search(self):
        self.data["message"] = "jogging sneakers"
        result = app.run(self.data)
        self.assertEqual(result["search"]["results"][0]["id"], "synthetic-runner")
        self.assertEqual(result["search"]["results"][0]["matched_terms"], ["running", "shoe"])

    def test_cross_stage_propagation(self):
        result = app.run(self.data)
        request = result["support"]["search_request"]
        self.assertEqual(request["query"], result["search"]["query"])
        self.assertEqual(request["filters"], result["search"]["filters"])
        self.assertEqual(result["search"]["filters"], self.data["filters"])

    def test_no_answer_still_searches(self):
        self.data["message"] = "sneakers"
        result = app.run(self.data)
        self.assertTrue(result["support"]["needs_human"])
        self.assertEqual(result["support"]["answer"], app.FALLBACK)
        self.assertTrue(result["search"]["results"])

    def test_constraints_and_availability(self):
        result = app.run(self.data)["search"]["results"]
        self.assertEqual({r["id"] for r in result}, {"synthetic-runner", "synthetic-walker"})
        self.data["filters"]["max_price"] = 0
        self.assertEqual(app.run(self.data)["search"]["results"], [])

    def test_category_and_limit(self):
        self.data["filters"]["limit"] = 1
        self.assertEqual(len(app.run(self.data)["search"]["results"]), 1)
        self.data["filters"]["category"] = "Electronics"
        self.assertEqual(app.run(self.data)["search"]["results"], [])

    def test_empty_catalog_and_knowledge(self):
        self.data["knowledge"] = []
        self.data["products"] = []
        result = app.run(self.data)
        self.assertTrue(result["support"]["needs_human"])
        self.assertEqual(result["search"]["results"], [])

    def test_nonmatching_and_stopword_queries(self):
        for query in ("the and please", "quantum telescope", "!!!", "你好"):
            with self.subTest(query=query):
                self.data["message"] = query
                self.assertEqual(app.run(self.data)["search"]["results"], [])

    def test_invalid_inputs(self):
        mutations = [
            lambda d: d.update(message=" "),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(extra=1),
            lambda d: d["filters"].update(max_price=float("nan")),
            lambda d: d["filters"].update(limit=True),
            lambda d: d["products"][0].update(price=-1),
            lambda d: d["products"][0].update(in_stock="yes"),
            lambda d: d["products"].append(copy.deepcopy(d["products"][0])),
            lambda d: d["knowledge"][0].update(keywords="returns"),
        ]
        for mutation in mutations:
            data = copy.deepcopy(self.data)
            mutation(data)
            with self.subTest(data=data), self.assertRaises(app.ValidationError):
                app.run(data)

    def test_rejects_tampered_handoff(self):
        support = app.support_stage(self.data)
        support["search_request"]["filters"]["max_price"] = 1000
        with self.assertRaises(app.ValidationError):
            app.search_stage(support, self.data)

    def test_injected_callable_contract(self):
        self.assertEqual(app.run(self.data, lambda proposal: proposal), app.run(self.data))
        def ungrounded(proposal):
            proposal["answer"] = "Everything is free."
            return proposal
        with self.assertRaises(app.ValidationError):
            app.run(self.data, ungrounded)
        with self.assertRaises(app.ValidationError):
            app.run(self.data, lambda proposal: None)

    def test_deterministic_and_no_input_mutation(self):
        original = copy.deepcopy(self.data)
        first = app.run(self.data)
        self.data["products"].reverse()
        self.assertEqual(first, app.run(self.data))
        self.data["products"].reverse()
        self.assertEqual(self.data, original)

    def test_multiple_support_topics(self):
        self.data["message"] = "shipping and returns"
        result = app.run(self.data)["support"]
        self.assertEqual(set(result["citations"]), {"synthetic-return", "synthetic-delivery"})

    def test_cli_success(self):
        result = self.cli(ROOT / "example_input.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertEqual(result.stderr, "")

    def test_cli_missing_and_invalid_file(self):
        for args in ((), (ROOT / "does-not-exist.json",), (ROOT / "implementation.py",)):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "error")
                self.assertEqual(result.stderr, "")

    def test_strict_json_helpers(self):
        with self.assertRaises(app.ValidationError):
            json.loads('{"x":1,"x":2}', object_pairs_hook=app.unique_object)
        with self.assertRaises(app.ValidationError):
            json.loads('NaN', parse_constant=app.reject_constant)


if __name__ == "__main__":
    unittest.main()
