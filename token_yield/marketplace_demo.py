"""Offline security-document demo with real ridge fits on synthetic measurements.

No network, language model, shell, or authoritative compliance assessment is used.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from token_yield.robust import Record, RidgeLinearModel

RATES = {"gpt-mini": (0.25, 2.0), "gpt": (1.25, 10.0),
         "claude": (3.0, 15.0), "gemini": (0.3, 2.5)}
BASE_FEATURES = (
    "document_units", "wiki_links", "requirements", "evidence_quotes",
    "gaps", "description_units", "offline_source_reviews",
)
STAGES = ("decompose", "wiki", "requirements", "features", "predict_before",
          "simulate", "train", "evaluate", "predict_after", "complete")
LIMITATIONS = [
    "All token measurements are explicitly SYNTHETIC, not observed AI usage.",
    "Real local feature engineering and ridge training; no internet or LLM calls.",
    "Deep research means offline review of the bundled fictional sources only.",
    "Ground truth is the fictional demo reference baseline, not authoritative compliance.",
    "Evidence matching uses explicit demo tags, not semantic or legal judgment.",
    "Synthetic holdout MAE does not establish real-world prediction accuracy.",
    "Model prices are illustrative scenario rates, not current vendor quotes.",
    "Estimates cover ONE entire project execution; monthly cost multiplies executions.",
    "Submitted descriptions configure this sample project; no user documents are ingested.",
]


class PipelineCancelled(Exception):
    """An explicit cancellation stopped execution before the next stage."""


def validate_request(value: Any) -> dict[str, Any]:
    """Validate and normalize the fixed HTTP and synchronous pipeline contract."""
    if not isinstance(value, dict):
        raise ValueError("body must be a JSON object")
    if set(value) - {"description", "model_id", "runs_per_month", "new_function", "execution_mode"}:
        raise ValueError("unknown request fields")
    description = value.get("description")
    if not isinstance(description, str) or not 20 <= len(description.strip()) <= 6000:
        raise ValueError("description must contain 20..6000 characters")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or model_id not in RATES:
        raise ValueError("model_id must be gpt-mini, gpt, claude, or gemini")
    runs = value.get("runs_per_month")
    if type(runs) is not int or not 0 <= runs <= 1_000_000:
        raise ValueError("runs_per_month must be an integer from 0 to 1000000")
    novel = value.get("new_function", "")
    if not isinstance(novel, str) or len(novel) > 120:
        raise ValueError("new_function must be a string of at most 120 characters")
    mode = value.get("execution_mode", "automatic")
    if not isinstance(mode, str) or mode not in ("step", "automatic"):
        raise ValueError("execution_mode must be step or automatic")
    # Reject lone surrogate characters before creating filesystem artifacts.
    try:
        (description + novel).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("text must be valid UTF-8") from exc
    return {"description": description.strip(), "model_id": model_id,
            "runs_per_month": runs, "new_function": novel.strip(), "execution_mode": mode}


def sample_documents() -> list[dict[str, Any]]:
    """Return original fictional source material with intentional evidence gaps."""
    sources = [
        ("demo-reference", "Fictional Harbor reference baseline", "reference",
         "Original fictional demo baseline, not a regulation or certification.\n"
         "REQ-1 | Restrict privileged access and review it quarterly. | access,access-review\n"
         "REQ-2 | Encrypt customer data in transit and at rest. | tls,at-rest\n"
         "REQ-3 | Assign an incident owner and test response annually. | incident-owner,incident-test\n"
         "REQ-4 | Document retention periods and verified deletion. | retention,deletion\n"
         "REQ-5 | Test restoration and record the result. | restore-test\n"
         "REQ-6 | Log administrative actions and review alerts. | audit-log,alert-review\n"
         "Review project sources [[harbor-security]] and [[harbor-operations]]."),
        ("harbor-security", "Harbor security design", "project",
         "Fictional Harbor is an internal document service.\n"
         "[access] Privileged access is limited to named administrators.\n"
         "[access-review] The security owner reviews administrator access quarterly.\n"
         "[tls] All client connections use TLS.\n"
         "Encryption at rest has not yet been documented.\n"
         "Operations evidence is maintained in [[harbor-operations]]."),
        ("harbor-operations", "Harbor operations handbook", "project",
         "[incident-owner] The on-call lead owns incident coordination.\n"
         "An annual incident exercise is planned but has not occurred.\n"
         "[audit-log] Administrator actions are written to the audit log.\n"
         "[alert-review] The on-call engineer reviews administrative alerts daily.\n"
         "No restoration test record is available.\n"
         "See architecture [[harbor-security]] and decisions [[harbor-decisions]]."),
        ("harbor-decisions", "Harbor decision log", "project",
         "Retention and deletion requirements remain unresolved.\n"
         "The demo reference is [[demo-reference]]. No live customer data is present."),
    ]
    ids = {source[0] for source in sources}
    return [
        {"id": doc_id, "title": title, "text": text, "role": role,
         "links": [link for link in re.findall(r"\[\[([\w-]+)\]\]", text) if link in ids]}
        for doc_id, title, role, text in sources
    ]


def extract_requirements(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract baseline rows and quote only project lines bearing evidence tags."""
    requirements = []
    for source in documents:
        if source["role"] != "reference":
            continue
        for line in source["text"].splitlines():
            match = re.fullmatch(r"(REQ-\d+) \| (.+) \| ([\w,-]+)", line)
            if not match:
                continue
            req_id, text, tag_text = match.groups()
            tags = tag_text.split(",")
            evidence, found = [], set()
            for doc in documents:
                if doc["role"] != "project":
                    continue
                for quote in doc["text"].splitlines():
                    for tag in tags:
                        if quote.startswith(f"[{tag}] "):
                            evidence.append({"document_id": doc["id"], "quote": quote})
                            found.add(tag)
            status = "evidence_found" if len(found) == len(tags) else "partial" if found else "missing"
            requirements.append({"id": req_id, "text": text, "source_id": source["id"],
                                 "status": status, "evidence": evidence})
    return requirements


def decompose(request: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    catalog = [
        ("ingest", "Document ingestion", ["load fictional documents", "index source text"]),
        ("wiki", "Wiki linking", ["resolve local wiki links", "retain source provenance"]),
        ("requirements", "Reference requirements extraction", ["parse demo baseline controls"]),
        ("evidence", "Evidence mapping", ["match explicit evidence tags", "quote project sources"]),
        ("gaps", "Gap review", ["compare required tags", "list missing evidence"]),
        ("report", "Report", ["summarize demo findings", "estimate whole-project cost"]),
    ]
    bricks = [{"id": key, "name": name, "atoms": atoms, "novel": False}
              for key, name, atoms in catalog]
    if re.search(r"\b(deep research|source review)\b", request["description"], re.I):
        bricks.insert(-1, {"id": "source-review", "name": "Offline source review",
                           "atoms": ["review bundled reference and project sources"], "novel": False})
    novel = request["new_function"]
    if not novel and re.search(r"\bpolicy[\s-]+as[\s-]+code\b", request["description"], re.I):
        novel = "Policy-as-code parsing"
    if novel:
        key = "novel_task_" + hashlib.sha256(novel.casefold().encode()).hexdigest()[:12]
        bricks.insert(-1, {"id": key, "name": novel,
                          "atoms": ["add requested task to synthetic workload feature schema"],
                          "novel": True})
        return bricks, key
    return bricks, None


def _samples(names: list[str], seed: int, count: int, prefix: str) -> list[dict[str, Any]]:
    """Generate whole-project synthetic observations; one independent group per row."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        values = [rng.uniform(100, 1800), rng.randint(1, 18), rng.randint(3, 24),
                  rng.randint(0, 30), rng.randint(0, 14), rng.uniform(5, 1600),
                  rng.randint(0, 2)]
        if len(names) > len(BASE_FEATURES):
            values.append(rng.randint(0, 5))
        novel = values[7] if len(values) > 7 else 0
        input_tokens = (250 + 2.7 * values[0] + 35 * values[1] + 90 * values[2]
                        + 48 * values[3] + 72 * values[4] + 1.3 * values[5]
                        + 420 * values[6] + 1100 * novel + rng.uniform(-90, 90))
        output_tokens = (110 + .18 * values[0] + 12 * values[1] + 60 * values[2]
                         + 32 * values[3] + 95 * values[4] + .08 * values[5]
                         + 220 * values[6] + 480 * novel + rng.uniform(-45, 45))
        rows.append({"split": "test" if index % 5 == 0 else "train",
                     "group": f"{prefix}-{index}", "features": dict(zip(names, values)),
                     "input_tokens": round(input_tokens, 3),
                     "output_tokens": round(output_tokens, 3), "synthetic": True,
                     "unit": "one entire project execution"})
    return rows


def _fit(rows: list[dict[str, Any]], names: list[str]) -> dict[str, RidgeLinearModel]:
    return {
        channel: RidgeLinearModel.fit([
            Record(tuple(row["features"][name] for name in names),
                   row[f"{channel}_tokens"], row["group"])
            for row in rows if row["split"] == "train"
        ], alpha=0.5)
        for channel in ("input", "output")
    }


def _coefficients(models: dict[str, RidgeLinearModel], names: list[str]) -> dict[str, Any]:
    return {channel: {"intercept": model.raw_intercept,
                      **dict(zip(names, model.raw_coefficients))}
            for channel, model in models.items()}


def _predict(models: dict[str, RidgeLinearModel], names: list[str],
             features: dict[str, float]) -> dict[str, float]:
    predictions = {f"{channel}_tokens": round(max(0, model.predict(
        [features[name] for name in names]))) for channel, model in models.items()}
    predictions["total_tokens"] = sum(predictions.values())
    return predictions


def write_json(path: Path, value: Any) -> None:
    """Persist JSON atomically in the run's own directory."""
    staging = path.with_suffix(path.suffix + ".writing")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    staging.replace(path)


def run_pipeline(request: dict[str, Any], run_dir: Path, *,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 before_stage: Callable[[str], None] | None = None,
                 run_id: str | None = None) -> dict[str, Any]:
    """Execute the offline pipeline synchronously and persist reproducible artifacts.

    Args:
        request: Validated description, scenario model, volume, optional novel task.
        run_dir: Trusted CLI-selected storage root, never taken from HTTP input.
        on_event: Optional callback invoked once per append-only event.
        before_stage: Called before each stage's actual work; may block or cancel.
        run_id: Internal UUID hex identifier; a new UUID is generated by default.

    Returns:
        The final API result, with fitted estimates and synthetic holdout metrics.
    """
    request = validate_request(request)
    if request["execution_mode"] == "step" and before_stage is None:
        raise ValueError("step execution requires a before_stage approval callback")
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("run_id must be a UUID hex identifier")
    folder = Path(run_dir) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", request)
    events: list[dict[str, Any]] = []

    def begin(stage: str) -> None:
        if before_stage:
            before_stage(stage)

    def emit(stage: str, message: str, data: dict[str, Any]) -> None:
        event = {"seq": len(events) + 1, "time": datetime.now(timezone.utc).isoformat(),
                 "stage": stage, "message": message, "data": data}
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        events.append(event)
        if on_event:
            on_event(event)

    try:
        begin("decompose")
        bricks, novel_key = decompose(request)
        emit("decompose", "Decomposed the fictional project into local workflow bricks.",
             {"bricks": bricks})
        begin("wiki")
        documents = sample_documents()
        emit("wiki", "Ingested original fictional sources and resolved local wiki links.",
             {"documents": documents})
        begin("requirements")
        requirements = extract_requirements(documents)
        emit("requirements", "Compared project evidence with the demo reference, not a compliance standard.",
             {"requirements": requirements})
        begin("features")
        features = dict(zip(BASE_FEATURES, [
            sum(len(doc["text"]) for doc in documents) / 4,
            sum(len(doc["links"]) for doc in documents), len(requirements),
            sum(len(req["evidence"]) for req in requirements),
            sum(req["status"] != "evidence_found" for req in requirements),
            len(request["description"]) / 4,
            int(any(brick["id"] == "source-review" for brick in bricks)),
        ]))
        if novel_key:
            features[novel_key] = 1
        names = list(features)
        emit("features", "Engineered numeric features; length/4 is a proxy, not a tokenizer.",
             {"features": features, "feature_names": names})
        begin("predict_before")
        base_names = list(BASE_FEATURES)
        baseline_rows = _samples(base_names, 47, 100, "catalog")
        baseline_models = _fit(baseline_rows, base_names)
        if novel_key:
            before = {"supported": False, "reason": f"Missing catalog feature: {novel_key}",
                      "input_tokens": None, "output_tokens": None, "total_tokens": None}
        else:
            before = {"supported": True, "reason": "All features supported by fitted synthetic catalog.",
                      **_predict(baseline_models, base_names, features)}
        before["coefficients"] = _coefficients(baseline_models, base_names)
        emit("predict_before", "Generated SYNTHETIC baseline catalog, fitted it, then predicted or abstained.", before)
        begin("simulate")
        # Existing catalog observations are retained with zero novel-task exposure.
        for row in baseline_rows:
            if novel_key:
                row["features"][novel_key] = 0
        rows = baseline_rows + _samples(names, 193, 150, "presimulation")
        emit("simulate", "Generated explicitly SYNTHETIC whole-project measurements locally.",
             {"source": "synthetic", "count": len(rows), "unit": "one entire project execution",
              "generator_seeds": [47, 193], "novel_feature": novel_key,
              "rows_preview": rows[:5] + rows[100:105]})
        begin("train")
        models = _fit(rows, names)
        coefficients = _coefficients(models, names)
        emit("train", "Actually fitted two RidgeLinearModel regressors on training groups only.",
             {"coefficients": coefficients, "alpha": 0.5, "source": "synthetic"})
        begin("evaluate")
        heldout = [row for row in rows if row["split"] == "test"]
        metrics = {
            f"{channel}_mae": sum(abs(row[f"{channel}_tokens"] - model.predict(
                [row["features"][name] for name in names])) for row in heldout) / len(heldout)
            for channel, model in models.items()
        }
        training = {"version": "synthetic-ridge-v1-expanded" if novel_key else "synthetic-ridge-v1",
                    "train_count": len(rows) - len(heldout), "test_count": len(heldout),
                    **metrics, "rows": rows, "coefficients": coefficients, "source": "synthetic",
                    "split_unit": "independent synthetic project group"}
        emit("evaluate", "Evaluated disjoint synthetic test groups; NOT evidence of real-world accuracy.",
             {key: value for key, value in training.items() if key not in ("rows", "coefficients")})
        begin("predict_after")
        after = _predict(models, names, features)
        input_rate, output_rate = RATES[request["model_id"]]
        cost = (after["input_tokens"] * input_rate + after["output_tokens"] * output_rate) / 1_000_000
        after.update({"usd_per_run": cost, "usd_per_month": cost * request["runs_per_month"]})
        emit("predict_after", "Re-predicted ONE entire project execution using the newly fitted model.",
             {"after": after, "rates_illustrative": True})
        begin("complete")
        result = {"id": run_id, "bricks": bricks, "documents": documents,
                  "requirements": requirements, "features": features, "feature_names": names,
                  "training": training, "before": before, "after": after,
                  "rates": {"input_per_million": input_rate, "output_per_million": output_rate,
                            "illustrative": True},
                  "limitations": LIMITATIONS,
                  "report": {"baseline": "fictional demo reference",
                             "counts": {status: sum(req["status"] == status for req in requirements)
                                        for status in ("evidence_found", "partial", "missing")}}}
        write_json(folder / "training.json", training)
        write_json(folder / "model.json", {
            "feature_names": names, "source": "synthetic",
            "models": {channel: asdict(model) for channel, model in models.items()},
            "baseline_coefficients": before["coefficients"], "coefficients": coefficients,
        })
        write_json(folder / "result.json", result)
        emit("complete", "Offline run completed; result, fitted model, and rows saved locally.",
             {"id": run_id, "source": "synthetic"})
        return result
    except Exception as exc:
        write_json(folder / "error.json", {
            "error": str(exc), "status": "cancelled" if isinstance(exc, PipelineCancelled) else "failed",
        })
        raise
