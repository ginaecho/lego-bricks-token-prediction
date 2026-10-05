"""Synthetic telecom reference pipeline. Python standard library; no providers."""
import csv
import io
import json
import math
import re
import sys
from collections import Counter


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def exact(obj, keys, label):
    require(isinstance(obj, dict) and set(obj) == set(keys), label + " has invalid fields")


COMMON = {"id", "kind", "account_id", "residency", "synthetic"}
FIELDS = {
    "subscriber": {"consent", "purpose", "phone", "imei"},
    "ticket": {"description", "state"},
    "usage": {"call_seconds", "data_mb", "phone", "imei"},
    "chat": {"text"},
    "adjustment": {"amount_cents", "reason"},
}
CSV_FIELDS = ["id", "account_id", "residency", "synthetic",
              "call_seconds", "data_mb", "phone", "imei"]
ACTIONS = {
    "diagnose_network": ("network", []),
    "request_network_repair": ("network", ["diagnose_network"]),
    "review_usage": ("usage", []),
    "request_usage_guidance": ("usage", ["review_usage"]),
    "inspect_bill": ("billing", []),
    "propose_billing_adjustment": ("billing", ["inspect_bill"]),
}
ALIASES = {
    "internet": "network", "signal": "network", "outage": "network",
    "dropped": "network", "disconnect": "network", "connectivity": "network",
    "fault": "network", "repair": "network", "slow": "network",
    "data": "usage", "calls": "usage", "volume": "usage",
    "charge": "billing", "bill": "billing", "refund": "billing",
    "invoice": "billing", "charged": "billing",
}


def redact(value):
    value = re.sub(r"SYN-IMEI-[A-Za-z0-9-]+", "[DEVICE]", value, flags=re.I)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    value = re.sub(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)", "[PHONE_OR_DEVICE]", value)
    return value


def tokens(value):
    return [ALIASES.get(word, word) for word in re.findall(r"[a-z]+", value.lower())]


def validate_records(records, residency):
    require(isinstance(records, list) and records, "records must be a nonempty list")
    seen = set()
    accounts = set()
    for record in records:
        require(isinstance(record, dict), "record must be an object")
        kind = record.get("kind")
        require(isinstance(kind, str) and kind in FIELDS, "unknown record kind")
        exact(record, COMMON | FIELDS[kind], "record")
        identifier = text(record["id"], "id")
        require(re.fullmatch(r"[A-Z][A-Z0-9_-]{1,39}", identifier), "invalid record id")
        require(identifier not in seen, "duplicate record id")
        seen.add(identifier)
        require(isinstance(record["account_id"], str)
                and re.fullmatch(r"ACC-[A-Z0-9-]+", record["account_id"]),
                "account_id must be a pseudonymous ACC identifier")
        require(record["residency"] == residency, "record residency mismatch")
        require(record["synthetic"] is True, "only labeled synthetic records accepted")
        if kind == "subscriber":
            require(record["id"] == record["account_id"], "subscriber id must match account_id")
            require(record["consent"] is True and record["purpose"] == "service_support",
                    "subscriber consent and service_support purpose required")
            accounts.add(record["id"])
        if kind in ("subscriber", "usage"):
            require(isinstance(record["phone"], str)
                    and re.fullmatch(r"\+1-202-555-01\d{2}", record["phone"]),
                    "phone must use fictitious reserved range")
            require(isinstance(record["imei"], str)
                    and re.fullmatch(r"SYN-IMEI-\d{8}", record["imei"]),
                    "device must be a synthetic IMEI marker")
        if kind == "ticket":
            text(record["description"], "description")
            require(record["state"] in ("open", "resolved"), "invalid ticket state")
        if kind == "chat":
            text(record["text"], "chat text")
        if kind == "usage":
            for field in ("call_seconds", "data_mb"):
                require(type(record[field]) is int and 0 <= record[field] <= 10**12,
                        field + " must be a bounded nonnegative integer")
        if kind == "adjustment":
            require(type(record["amount_cents"]) is int
                    and 0 < abs(record["amount_cents"]) <= 10**9, "invalid adjustment amount")
            text(record["reason"], "billing adjustment reason")
    require(all(record["account_id"] in accounts for record in records),
            "record references unknown subscriber")


def parse_input(data):
    exact(data, {"schema_version", "synthetic", "processing_residency", "account_id",
                 "query", "completed_actions", "records", "cdr_csv"}, "input")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "unsupported schema version")
    require(data["synthetic"] is True, "input must be explicitly synthetic")
    require(data["processing_residency"] in ("EU", "US"), "unsupported residency")
    text(data["query"], "query")
    require(len(data["query"]) <= 1000, "query too long")
    completed = data["completed_actions"]
    require(isinstance(completed, list) and all(isinstance(x, str) and x in ACTIONS
                                               for x in completed), "invalid completed actions")
    require(len(set(completed)) == len(completed), "duplicate completed action")
    require(all(set(ACTIONS[x][1]) <= set(completed) for x in completed),
            "completed action missing prerequisite")
    require(isinstance(data["records"], list), "records must be a list")
    require(isinstance(data["cdr_csv"], str), "cdr_csv must be text")
    records = [dict(record) if isinstance(record, dict) else record for record in data["records"]]
    try:
        reader = csv.DictReader(io.StringIO(data["cdr_csv"]), strict=True)
        require(reader.fieldnames == CSV_FIELDS, "invalid CDR CSV header")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    "malformed CDR CSV row")
            require(row["synthetic"] == "true", "CDR must be synthetic")
            for field in ("call_seconds", "data_mb"):
                require(re.fullmatch(r"\d{1,13}", row[field]), "invalid CDR numeric value")
                row[field] = int(row[field])
            row["kind"] = "usage"
            row["synthetic"] = True
            records.append(row)
    except csv.Error as exc:
        raise ValidationError("invalid CDR CSV") from exc
    validate_records(records, data["processing_residency"])
    require(any(r["kind"] == "subscriber" and r["id"] == data["account_id"] for r in records),
            "selected subscriber not found")
    selected = [r for r in records if r["account_id"] == data["account_id"]]
    safe = []
    for record in selected:
        if record["kind"] == "subscriber":
            content = "Synthetic subscriber authorized for service support"
        elif record["kind"] == "ticket":
            content = record["description"]
        elif record["kind"] == "chat":
            content = record["text"]
        elif record["kind"] == "usage":
            content = "Call and data usage record"
        else:
            content = "Billing adjustment: " + record["reason"]
        safe.append({"id": record["id"], "kind": record["kind"],
                     "account_id": record["account_id"], "residency": record["residency"],
                     "synthetic": True, "text": redact(content),
                     "values": {k: redact(v) if isinstance(v, str) else v
                                for k, v in record.items()
                                if k in {"call_seconds", "data_mb", "amount_cents", "reason", "state"}}})
    envelope = {"schema_version": 1, "synthetic": True, "status": "ok", "stage": "input",
                "account_id": data["account_id"], "residency": data["processing_residency"],
                "query": redact(data["query"]), "completed_actions": list(completed),
                "records": safe, "search": [], "journey": [], "documents": [], "feedback": []}
    validate_envelope(envelope, "input")
    return envelope


def validate_envelope(env, stage):
    exact(env, {"schema_version", "synthetic", "status", "stage", "account_id", "residency",
                "query", "completed_actions", "records", "search", "journey", "documents",
                "feedback"}, "pipeline envelope")
    require(env["stage"] == stage and env["status"] == "ok", "wrong pipeline stage")
    require(type(env["schema_version"]) is int and env["schema_version"] == 1
            and env["synthetic"] is True, "invalid envelope metadata")
    require(env["residency"] in ("EU", "US"), "invalid envelope residency")
    require(isinstance(env["records"], list) and env["records"], "missing source records")
    require(isinstance(env["completed_actions"], list)
            and all(isinstance(a, str) and a in ACTIONS for a in env["completed_actions"]),
            "invalid completed actions")
    require(len(set(env["completed_actions"])) == len(env["completed_actions"])
            and all(set(ACTIONS[a][1]) <= set(env["completed_actions"])
                    for a in env["completed_actions"]), "invalid completed prerequisites")
    ids = set()
    for record in env["records"]:
        exact(record, COMMON | {"text", "values"}, "safe record")
        require(record["id"] not in ids, "duplicate safe record id")
        ids.add(record["id"])
        require(record["residency"] == env["residency"]
                and record["account_id"] == env["account_id"] and record["synthetic"] is True,
                "source record boundary violation")
        require(record["kind"] in FIELDS and isinstance(record["values"], dict),
                "invalid safe record")
        text(record["text"], "safe text")
        require(redact(record["text"]) == record["text"], "unredacted source text")
        value_fields = {"subscriber": set(), "ticket": {"state"}, "chat": set(),
                        "usage": {"call_seconds", "data_mb"},
                        "adjustment": {"amount_cents", "reason"}}
        exact(record["values"], value_fields[record["kind"]], "safe values")
        if record["kind"] == "usage":
            require(all(type(record["values"][k]) is int
                        and 0 <= record["values"][k] <= 10**12
                        for k in ("call_seconds", "data_mb")), "invalid safe usage")
        if record["kind"] == "ticket":
            require(record["values"]["state"] in ("open", "resolved"), "invalid safe ticket")
        if record["kind"] == "adjustment":
            text(record["values"].get("reason"), "adjustment reason")
            amount = record["values"]["amount_cents"]
            require(type(amount) is int and 0 < abs(amount) <= 10**9,
                    "invalid safe adjustment")
            require(redact(record["values"]["reason"]) == record["values"]["reason"],
                    "unredacted adjustment reason")
    for key in ("search", "journey", "documents", "feedback"):
        require(isinstance(env[key], list), key + " must be a list")
    order = ["input", "search", "journey", "documents", "feedback"]
    require(stage in order, "unknown pipeline stage")
    for future in order[order.index(stage) + 1:]:
        require(not env[future], "unexpected future-stage output")
    search_ids = set()
    for item in env["search"]:
        exact(item, {"record_id", "score", "residency"}, "search result")
        require(item["record_id"] in ids and item["record_id"] not in search_ids,
                "invalid search reference")
        search_ids.add(item["record_id"])
        require(type(item["score"]) in (int, float) and math.isfinite(item["score"])
                and item["score"] > 0, "invalid search score")
        require(item["residency"] == env["residency"], "search residency mismatch")
    done = set(env["completed_actions"])
    for index, step in enumerate(env["journey"]):
        exact(step, {"id", "position", "prerequisites", "source_ids", "residency"}, "journey step")
        require(step["id"] in ACTIONS and step["id"] not in done, "invalid journey action")
        require(step["position"] == index + 1 and step["prerequisites"] == ACTIONS[step["id"]][1]
                and set(step["prerequisites"]) <= done, "journey prerequisite violation")
        require(isinstance(step["source_ids"], list) and step["source_ids"]
                and len(set(step["source_ids"])) == len(step["source_ids"])
                and set(step["source_ids"]) <= search_ids,
                "journey requires search evidence")
        require(step["residency"] == env["residency"], "journey residency mismatch")
        done.add(step["id"])
    if stage in ("journey", "documents", "feedback"):
        require(len(env["journey"]) == 2, "journey must have two steps")
    sources = {r["id"]: r for r in env["records"]}
    for index, doc in enumerate(env["documents"]):
        exact(doc, {"id", "action_id", "account_id", "residency", "synthetic",
                    "source_ids", "extracted", "checks"}, "document")
        require(index < len(env["journey"]), "extra document")
        step = env["journey"][index]
        require(doc["id"] == "DOC-" + str(index + 1) and doc["action_id"] == step["id"]
                and doc["source_ids"] == step["source_ids"], "document journey mismatch")
        require(doc["account_id"] == env["account_id"] and doc["residency"] == env["residency"]
                and doc["synthetic"] is True, "document boundary violation")
        require(doc["extracted"] == extract([sources[s] for s in doc["source_ids"]]),
                "document extraction mismatch")
        require(doc["checks"] == {"source_links_valid": True, "totals_valid": True,
                                   "billing_reasons_present": True}, "invalid document checks")
    if stage in ("documents", "feedback"):
        require(len(env["documents"]) == 2, "documents must cover journey")
    supported = {s for d in env["documents"] for s in d["source_ids"]}
    seen_themes = set()
    for theme in env["feedback"]:
        exact(theme, {"theme", "count", "residency", "evidence"}, "feedback theme")
        require(theme["residency"] == env["residency"], "feedback residency mismatch")
        require(theme["theme"] in ("network", "usage", "billing", "other"), "invalid theme")
        require(theme["theme"] not in seen_themes, "duplicate theme")
        seen_themes.add(theme["theme"])
        require(theme["count"] == len(theme["evidence"]) > 0, "invalid feedback count")
        seen_excerpts = set()
        for evidence in theme["evidence"]:
            exact(evidence, {"source_ids", "document_ids", "excerpt", "residency"}, "evidence")
            require(evidence["source_ids"] and set(evidence["source_ids"]) <= supported,
                    "untraceable feedback")
            require(evidence["document_ids"] == [d["id"] for d in env["documents"]
                    if set(evidence["source_ids"]) & set(d["source_ids"])],
                    "invalid supporting document links")
            require(evidence["residency"] == env["residency"], "evidence residency mismatch")
            require(all(sources[s]["kind"] == "chat"
                        and evidence["excerpt"] == " ".join(sources[s]["text"].split())[:240]
                        for s in evidence["source_ids"]), "invalid supporting excerpt")
            full_text = " ".join(sources[evidence["source_ids"][0]]["text"].split())
            require(full_text not in seen_excerpts, "duplicate feedback message")
            seen_excerpts.add(full_text)
            expected_ids = [r["id"] for r in env["records"] if r["kind"] == "chat"
                            and r["id"] in supported and " ".join(r["text"].split()) == full_text]
            require(evidence["source_ids"] == expected_ids, "incomplete duplicate provenance")
            found = set(tokens(full_text)) & {"network", "usage", "billing"}
            require(theme["theme"] in (found or {"other"}), "unsupported feedback theme")
    return env


def cosine(a, b):
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    return sum((x / norm_a) * (y / norm_b) for x, y in zip(a, b)) if norm_a and norm_b else 0.0


def semantic_search(env, embedding=None):
    validate_envelope(env, "input")
    candidates = [r for r in env["records"] if r["kind"] != "subscriber"]
    require(candidates, "no searchable records")
    texts = [env["query"]] + [r["text"] for r in candidates]
    if embedding is not None:
        try:
            vectors = embedding(list(texts))
        except Exception as exc:
            raise ValidationError("injected embedding failed") from exc
        require(isinstance(vectors, list) and len(vectors) == len(texts), "invalid embedding batch")
        require(all(isinstance(v, list) and v for v in vectors), "empty embedding vector")
        width = len(vectors[0])
        require(all(len(v) == width and all(type(n) in (int, float)
                    and math.isfinite(n) and abs(n) <= 1e100 for n in v)
                    for v in vectors), "invalid embedding dimensions or numbers")
    else:
        counts = [Counter(tokens(t)) for t in texts]
        vocabulary = sorted(set().union(*(set(c) for c in counts)))
        vectors = [[c[t] for t in vocabulary] for c in counts]
    results = []
    for record, vector in zip(candidates, vectors[1:]):
        score = round(cosine(vectors[0], vector), 8)
        if score > 0:
            results.append({"record_id": record["id"], "score": score,
                            "residency": env["residency"]})
    env["search"] = sorted(results, key=lambda r: (-r["score"], r["record_id"]))
    env["stage"] = "search"
    return validate_envelope(env, "search")


def recommend_journey(env):
    validate_envelope(env, "search")
    require(env["search"], "no relevant records for a supported journey")
    records = {r["id"]: r for r in env["records"]}
    priorities = Counter()
    for hit in env["search"]:
        for token in set(tokens(records[hit["record_id"]]["text"])):
            if token in ("network", "usage", "billing"):
                priorities[token] += hit["score"]
    ranked = sorted(ACTIONS, key=lambda a: (-priorities[ACTIONS[a][0]], list(ACTIONS).index(a)))
    done = set(env["completed_actions"])
    steps = []
    for _ in range(2):
        available = [a for a in ranked if a not in done and set(ACTIONS[a][1]) <= done]
        require(available, "not enough unfinished actions for a two-step journey")
        action = available[0]
        steps.append({"id": action, "position": len(steps) + 1,
                      "prerequisites": list(ACTIONS[action][1]),
                      "source_ids": [h["record_id"] for h in env["search"]],
                      "residency": env["residency"]})
        done.add(action)
    env["journey"] = steps
    env["stage"] = "journey"
    return validate_envelope(env, "journey")


def extract(records):
    """General extract/reshape/check operation over validated typed source records."""
    return {
        "record_count": len(records),
        "by_kind": dict(sorted(Counter(r["kind"] for r in records).items())),
        "call_seconds": sum(r["values"].get("call_seconds", 0) for r in records),
        "data_mb": sum(r["values"].get("data_mb", 0) for r in records),
        "ticket_ids": sorted(r["id"] for r in records if r["kind"] == "ticket"),
        "adjustments": [{"source_id": r["id"], "amount_cents": r["values"]["amount_cents"],
                         "reason": r["values"]["reason"], "residency": r["residency"],
                         "synthetic": True}
                        for r in records if r["kind"] == "adjustment"],
    }


def automate_documents(env):
    validate_envelope(env, "journey")
    sources = {r["id"]: r for r in env["records"]}
    env["documents"] = [
        {"id": "DOC-" + str(i + 1), "action_id": step["id"],
         "account_id": env["account_id"], "residency": env["residency"], "synthetic": True,
         "source_ids": list(step["source_ids"]),
         "extracted": extract([sources[s] for s in step["source_ids"]]),
         "checks": {"source_links_valid": True, "totals_valid": True,
                    "billing_reasons_present": True}}
        for i, step in enumerate(env["journey"])]
    env["stage"] = "documents"
    return validate_envelope(env, "documents")


def analyze_feedback(env):
    validate_envelope(env, "documents")
    supported = {s for d in env["documents"] for s in d["source_ids"]}
    groups = {}
    for record in env["records"]:
        if record["kind"] != "chat" or record["id"] not in supported:
            continue
        # Whitespace-only normalization preserves the words in supporting excerpts.
        key = " ".join(record["text"].split())
        groups.setdefault(key, []).append(record)
    themes = {}
    for group in groups.values():
        found = set(tokens(group[0]["text"])) & {"network", "usage", "billing"}
        ids = [r["id"] for r in group]
        for theme in sorted(found or {"other"}):
            themes.setdefault(theme, []).append({
                "source_ids": ids,
                "document_ids": [d["id"] for d in env["documents"]
                                 if set(ids) & set(d["source_ids"])],
                "excerpt": " ".join(group[0]["text"].split())[:240],
                "residency": env["residency"]})
    env["feedback"] = [{"theme": theme, "count": len(evidence),
                        "residency": env["residency"], "evidence": evidence}
                       for theme, evidence in sorted(themes.items())]
    env["stage"] = "feedback"
    return validate_envelope(env, "feedback")


def run(data, embedding=None):
    return analyze_feedback(automate_documents(recommend_journey(
        semantic_search(parse_input(data), embedding))))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle)
        output = run(data)
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else "invalid input or unreadable file"
        print(json.dumps({"status": "error", "message": message}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
