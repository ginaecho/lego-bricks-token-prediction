"""Assemble the agent-interaction graph for one brick composition.

Each stage reports its real input, output and the exact source functions that
produce it. Measured builds replay their recorded LLM-request telemetry;
unmeasured compositions stop honestly at "not yet measured".
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from examples import build_simulation_corpus as bsc
from examples import train_construction_model as tcm
from token_yield import build_simulations as bs
from token_yield import build_waves_v3 as v3

import foundry
from corpus import Corpus
from live_builder import LiveBuild
from sandbox import Workspace
from training import Trainer

APP_DIR = Path(__file__).resolve().parent

EDGES = [
    ("designer", "planner", "bricks + requirements"),
    ("planner", "briefing", "build spec"),
    ("briefing", "builder", "builder instructions"),
    ("builder", "workspace", "write / test / repair"),
    ("workspace", "builder", "tool results"),
    ("builder", "verifier", "4 deliverables"),
    ("verifier", "telemetry", "accepted build"),
    ("telemetry", "datapoint", "AIC token label"),
    ("planner", "datapoint", "pre-build features"),
    ("datapoint", "features", "record"),
    ("features", "splitter", "46-feature row"),
    ("splitter", "trainer", "train / validation groups"),
    ("trainer", "gate", "candidate model"),
    ("gate", "model", "promoted artifact"),
    ("model", "designer", "token forecast"),
]


class Pipeline:
    def __init__(self, root: Path, corpus: Corpus, trainer: Trainer):
        self.root, self.corpus, self.trainer = root, corpus, trainer

    def code(self, *functions) -> list[dict]:
        refs = []
        for function in functions:
            path = Path(inspect.getsourcefile(function)).resolve()
            lines, line = inspect.getsourcelines(function)
            if path.is_relative_to(self.root.resolve()):
                file = path.relative_to(self.root.resolve()).as_posix()
            else:
                file = "demo/datapoint-factory/" + path.relative_to(APP_DIR).as_posix()
            refs.append({"function": f"{function.__module__}.{function.__qualname__}", "file": file, "line": line,
                         "end_line": line + len(lines) - 1, "source": "".join(lines)})
        return refs

    def brief(self, parts: list[str], industry: str | None) -> str:
        """The exact builder instructions for this composition (staging path shown as a placeholder)."""
        spec = v3.build_spec_v3(parts, industry)
        return v3.instructions({"parts": parts, "industry": industry, "trial_id": spec["id"]},
                               f"<staging>/{spec['id']}")

    def run(self, parts: list[str], industry: str | None, build_id: str | None = None,
            live: dict | None = None, live_files: dict | None = None, brief: str | None = None) -> dict:
        spec = v3.build_spec_v3(parts, industry)
        matches = self.corpus.matches(spec["split_group"], parts)
        chosen = live or next((p for p in matches if p["id"] == build_id), matches[0] if matches else None)
        brief = brief or v3.instructions({"parts": parts, "industry": industry,
                                          "trial_id": chosen["id"] if chosen else spec["id"]},
                                         f"runs/<staging>/{chosen['id'] if chosen else spec['id']}")
        features = spec["input_features"]
        files = live_files if live else self.corpus.build_files(chosen["id"]) if chosen else {}
        stages = [
            self._designer(parts, industry),
            self._planner(spec),
            self._briefing(brief),
            *self._build_stages(chosen, files),
            self._datapoint(spec, chosen),
            self._features(features),
            self._splitter(spec, chosen),
            self._trainer(),
            self._gate(),
            self._model(spec, chosen),
        ]
        return {
            "selection": {"parts": parts, "industry": industry, "membership": spec["id"],
                          "level": features["composition_level"]},
            "measured_runs": [{"id": p["id"], "wave": p.get("wave") or "wave1",
                               "total_tokens": p["build_token_usage"]["total_tokens"],
                               "exact_order": (p["input_features"].get("parts") or p["input_features"]["types"]) == parts}
                              for p in matches],
            "replayed": chosen["id"] if chosen else None,
            "live": bool(live),
            "stages": stages, "edges": [{"from": a, "to": b, "label": label} for a, b, label in EDGES],
        }

    def _designer(self, parts, industry) -> dict:
        return {"id": "designer", "label": "Solution designer", "actor": "human", "status": "done",
                "summary": f"{len(parts)} brick(s)" + (f" · {industry} context" if industry else ""),
                "input": None, "output": {"ordered_parts": parts, "industry": industry},
                "code": [], "rule": "Parts come from distinct basic functionalities; order is the execution order."}

    def _planner(self, spec) -> dict:
        f = spec["input_features"]
        return {"id": "planner", "label": "Composition planner", "actor": "code", "status": "done",
                "summary": f"level {f['composition_level']} · {f['integration_edge_count']} handoff(s) · "
                           f"{f['planned_acceptance_case_count']} planned tests",
                "input": {"parts": f["parts"], "industry": f["industry"]},
                "output": {"id": spec["id"], "requirements": spec["requirements"],
                           "integration": spec["integration"], "split_group": spec["split_group"]},
                "code": self.code(v3.build_spec_v3, v3.level_of, v3.split_group),
                "rule": "Integrated builds are measured directly; never the sum of standalone builds."}

    def _briefing(self, brief) -> dict:
        return {"id": "briefing", "label": "Builder brief", "actor": "code", "status": "done",
                "summary": f"{len(brief.split())} words · sha256 {hashlib.sha256(brief.encode()).hexdigest()[:10]}",
                "input": {"template": v3.BUILDER_TEMPLATE_VERSION}, "output": {"instructions": brief},
                "code": self.code(v3.instructions), "rule": "The frozen template fingerprint must match before dispatch."}

    def _build_stages(self, point, files) -> list[dict]:
        if point is None:
            note = "No measured build of this membership exists yet. Dispatching an isolated builder would create it."
            return [
                {"id": "builder", "label": "Builder agent (LLM)", "actor": "llm", "status": "not_measured",
                 "summary": "not yet measured", "input": None, "output": None, "events": [],
                 "code": [], "rule": note},
                {"id": "workspace", "label": "Sandbox tools", "actor": "tool", "status": "not_measured",
                 "summary": "files · python · unittest", "input": None, "output": None, "code": [], "rule": note},
                {"id": "verifier", "label": "Runtime verifier", "actor": "code", "status": "not_measured",
                 "summary": "-", "input": None, "output": None, "code": self.code(bsc.import_build), "rule": note},
                {"id": "telemetry", "label": "Usage collector", "actor": "code", "status": "not_measured",
                 "summary": "-", "input": None, "output": None,
                 "code": self.code(bs.read_usage, bs.measured_usage),
                 "rule": "A missing usage record is an error; no LLM guess or text-length estimate replaces it."},
            ]
        usage, evidence = point["build_token_usage"], point["build_evidence"]
        live = point["origin"] == "live_foundry_demo_build"
        events = [{**e, "fresh_input": e["input_tokens"] - e["cache_read_tokens"] - e["cache_write_tokens"]}
                  for e in usage["events"]]
        tests = evidence["test_output"].strip().splitlines()
        accepted = evidence.get("accepted", evidence["tests_passed"])
        ran = next((line.split(" in ")[0] for line in tests if line.startswith("Ran ")), "tests")
        return [
            {"id": "builder", "label": "Builder agent (LLM)", "actor": "llm", "status": "done",
             "summary": f"{usage['builder_models'][0]} · {usage['provider_call_count']} LLM requests",
             "input": {"agent_id": usage["builder_agent_id"], "model": usage["builder_models"],
                       **({"endpoint": usage["endpoint"], "deployment": usage["deployment"]} if live else {})},
             "output": {"provider_call_count": usage["provider_call_count"],
                        **({"builder_report": evidence.get("builder_report")} if live else {})}, "events": events,
             "code": self.code(LiveBuild.loop, foundry.complete) if live else [], "rule": usage["measurement_scope"]},
            {"id": "workspace", "label": "Sandbox tools", "actor": "tool", "status": "done",
             "summary": f"{len(files)} files written", "input": None,
             "output": {"files": files}, "code": self.code(Workspace.write, Workspace.run_tests) if live else [],
             "rule": ("Dedicated local directory; only the four deliverables; timeouts; no credentials in the environment."
                      if live else "Isolated empty directory; Python stdlib only; no network, git or sub-agents.")},
            {"id": "verifier", "label": "Runtime verifier", "actor": "code", "status": "done" if accepted else "failed",
             "summary": f"{ran} · {'OK' if accepted else 'FAILED'}",
             "input": {"test_command": evidence["test_command"], "cli_arguments": evidence["cli_arguments"]},
             "output": {"tests_passed": evidence["tests_passed"], "test_summary": tests[-3:],
                        "cli_exit_code": evidence["cli_exit_code"], "artifact_sha256": evidence["artifact_sha256"]},
             "code": self.code(Workspace.verify if live else bsc.import_build),
             "rule": "A build is accepted only if its tests and CLI pass again here."},
            {"id": "telemetry", "label": "Usage collector", "actor": "code", "status": "done",
             "summary": f"{usage['total_tokens']:,} tokens",
             "input": {"source": usage["source"], "events": len(events)},
             "output": {k: usage[k] for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                                              "cache_write_tokens", "total_tokens")},
             "code": self.code(foundry.complete) if live else self.code(bs.read_usage, bs.measured_usage),
             "rule": "AIC_total = sum(input) + sum(output); input already includes cache reads and writes."},
        ]

    def _datapoint(self, spec, point) -> dict:
        if point is None:
            return {"id": "datapoint", "label": "Data point", "actor": "artifact", "status": "not_measured",
                    "summary": "planned_unmeasured", "input": None,
                    "output": {"id": spec["id"], "status": "planned_unmeasured", "build_token_usage": None},
                    "code": self.code(v3.trial_point), "rule": "Unmeasured records are excluded from training."}
        usage = point["build_token_usage"]
        live = point["origin"] == "live_foundry_demo_build"
        return {"id": "datapoint", "label": "Data point", "actor": "artifact",
                "status": "done" if point["status"] == "measured_build_passed" else "failed",
                "summary": f"{point['id']}",
                "input": None,
                "output": {"id": point["id"], "status": point["status"], "wave": point.get("wave") or ("live" if live else "wave1"),
                           "split": point["split"], "label_input_tokens": usage["input_tokens"],
                           "label_output_tokens": usage["output_tokens"], "label_total_tokens": usage["total_tokens"],
                           **({"saved_to": f"demo/datapoint-factory/live_runs/{point['id']}/data_point.json"} if live else {}),
                           "azure_cost_scenarios": [{k: s[k] for k in ("target_model", "no_cache_estimate_usd",
                                                                       "observed_cache_profile_estimate_usd")}
                                                    for s in point.get("azure_cost_scenarios", [])]},
                "code": self.code(LiveBuild.point if live else bs.complete_point),
                "rule": point.get("training_use") or "Costs and outcomes are derived later; never model inputs."}

    def _features(self, features) -> dict:
        row = self.trainer.vector(features)
        active = [item for item in row if item["value"]]
        return {"id": "features", "label": "Feature engineer", "actor": "code", "status": "done",
                "summary": f"{len(row)} numeric features · {len(active)} non-zero",
                "input": {k: features[k] for k in ("parts", "industry", "composition_level")},
                "output": {"vector": row, "excluded": ["staff_*", "estimated_months_to_finish",
                                                       "observed test counts", "file sizes", "costs", "outcomes"]},
                "code": self.code(Trainer.vector, tcm.feature_names, tcm.matrix),
                "rule": "Only pre-build features; anything observed after construction would leak the label."}

    def _splitter(self, spec, point) -> dict:
        live = point is not None and point["origin"] == "live_foundry_demo_build"
        split = "not pooled (live builder)" if live else point["split"] if point else spec["split"]
        counts = {k: len(v) for k, v in self.trainer.split.items()}
        return {"id": "splitter", "label": "Grouped splitter", "actor": "code", "status": "done",
                "summary": f"this membership → {split}",
                "input": {"split_group": spec["split_group"]}, "output": {"split": split, "corpus_counts": counts},
                "code": self.code(v3.split_group), "rule": "Repeated or reordered builds of one membership share a split."}

    def _trainer(self) -> dict:
        t = self.trainer
        return {"id": "trainer", "label": "Ridge trainer", "actor": "code", "status": "done",
                "summary": f"grouped CV → alpha {t.alpha:g}",
                "input": {"development_records": len(t.development), "features": len(t.names)},
                "output": {"alpha_search": t.search, "selected_alpha": t.alpha},
                "code": self.code(Trainer._fit, tcm.grouped_cv, tcm.fit_predict, tcm.model),
                "rule": "Separate log-space ridge models for input and output tokens."}

    def _gate(self) -> dict:
        v = self.trainer.validation
        return {"id": "gate", "label": "Validation gate", "actor": "code", "status": "done" if v["passed"] else "failed",
                "summary": f"MAPE {v['candidate']['mape']:.1%} · {'passed' if v['passed'] else 'failed'}",
                "input": {"gates": self.trainer.gates},
                "output": {"candidate": v["candidate"], "baseline": v["training_mean_baseline"], "passed": v["passed"]},
                "code": self.code(Trainer._validate, tcm.metrics), "rule": "Promote only if MAPE ≤ 20% and MAE beats the training mean."}

    def _model(self, spec, point) -> dict:
        features = spec["input_features"]
        output = {"forecast": self.trainer.predict(features),
                  "top_effects_input_tokens": self.trainer.contributions(features)}
        if point is not None:
            output["actual_total_tokens"] = point["build_token_usage"]["total_tokens"]
            if point["origin"] == "live_foundry_demo_build":
                output["live_builder"] = point["build_token_usage"]["deployment"]
            else:
                output["forecast_before_this_datapoint"] = self.trainer.without_group(spec["split_group"], features)
        return {"id": "model", "label": "Token model", "actor": "artifact", "status": "done",
                "summary": f"≈ {output['forecast']['total_tokens']:,.0f} tokens",
                "input": {"features": len(self.trainer.names)}, "output": output,
                "code": self.code(Trainer.predict, Trainer.contributions, tcm.interval_bounds),
                "rule": "Bounded Python CLI builds by one builder model; not enterprise delivery effort."}
