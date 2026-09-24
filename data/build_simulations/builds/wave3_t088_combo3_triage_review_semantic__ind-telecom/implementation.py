"""Synthetic telecom triage -> evidence review -> local semantic retrieval.

Run: python -B implementation.py example_input.json
Only the Python standard library is required. No network or provider is used.
The shared canonical envelope is validated at every stage boundary. Raw chat,
CDR identifiers, and documentary evidence remain confined to intake; outputs
contain pseudonymous references, controlled tags, aggregates, and evidence IDs.
These are demonstration safeguards, not GDPR/CCPA or certification claims.
"""

import copy
import csv
import io
import json
import math
import re
import sys
from collections import Counter


class ValidationError(ValueError):
    """A safe, content-free input or stage-boundary error."""


STAGES = ("intake", "triage", "review", "semantic")
ID = re.compile(r"[A-Z][A-Z0-9_-]{0,39}\Z")
TAG = re.compile(r"[a-z][a-z_]{0,39}\Z")
RESIDENCY = re.compile(r"[A-Z]{2}\Z")
CDR_FIELDS = (
    "id", "account_id", "residency", "phone", "imei", "call_seconds", "data_mb"
)
SYNONYMS = {
    "outage": "network", "disconnected": "network", "connectivity": "network",
    "signal": "network", "fault": "network", "internet": "network",
    "invoice": "billing", "bill": "billing", "charge": "billing",
    "refund": "billing", "credit": "billing",
    "missing": "gap", "incomplete": "gap", "gaps": "gap",
}
DEFAULT_CONFIG = {
    "categories": [
        {"name": "network", "keywords": ["outage", "signal", "internet"],
         "priority": 2, "team": "network_operations", "owner": "fault_desk",
         "required_evidence": ["fault_description", "diagnostics"]},
        {"name": "billing", "keywords": ["bill", "charge", "refund"],
         "priority": 3, "team": "billing_support", "owner": "billing_desk",
         "required_evidence": ["usage_record", "adjustment_reason"]},
        {"name": "general", "keywords": [], "priority": 4,
         "team": "subscriber_support", "owner": "support_desk",
         "required_evidence": ["subscriber_request"]},
    ],
    "fallback": "general",
    "urgent_keywords": ["emergency", "widespread"],
}


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= value.keys(), "Required field missing")
    require(value.keys() <= set(required) | set(optional), "Unknown field")


def text(value, label="text", maximum=20000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            "Invalid " + label)
    return value


def token(value, pattern=ID):
    require(isinstance(value, str) and bool(pattern.fullmatch(value)),
            "Invalid identifier or tag")
    return value


def number(value, low=0, high=1e12, integer=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high and (not integer or type(value) is int),
            "Invalid numeric value")
    return value


def sequence(value, maximum=10000):
    require(isinstance(value, list) and len(value) <= maximum, "Invalid record list")
    return value


def unique(items):
    require(len(items) == len(set(items)), "Duplicate identifier")


def words(value):
    return re.findall(r"[a-z]+", value.lower())


def validate_config(config):
    keys(config, ("categories", "fallback", "urgent_keywords"))
    categories = sequence(config["categories"], 30)
    require(bool(categories), "At least one category is required")
    names = []
    for rule in categories:
        keys(rule, ("name", "keywords", "priority", "team", "owner", "required_evidence"))
        for field in ("name", "team", "owner"):
            token(rule[field], TAG)
        number(rule["priority"], 1, 4, integer=True)
        for field in ("keywords", "required_evidence"):
            values = sequence(rule[field], 50)
            for value in values:
                token(value, TAG)
            unique(values)
        names.append(rule["name"])
    unique(names)
    require(config["fallback"] in names, "Unknown fallback category")
    for value in sequence(config["urgent_keywords"], 50):
        token(value, TAG)
    return config


def validate_policy(policy):
    keys(policy, ("purpose", "allowed_residencies", "retention_days"))
    require(policy["purpose"] == "support_resolution", "Unsupported processing purpose")
    number(policy["retention_days"], 1, 365, integer=True)
    regions = sequence(policy["allowed_residencies"], 50)
    require(bool(regions), "Residency allowlist must not be empty")
    for region in regions:
        token(region, RESIDENCY)
    unique(regions)


def check_region(region, policy, expected=None):
    token(region, RESIDENCY)
    require(region in policy["allowed_residencies"], "Residency is not permitted")
    require(expected is None or region == expected, "Cross-residency relationship denied")


def canonical_validate(envelope, expected_stage=None):
    """The single validator for all pipeline handoffs and final output.

    Envelopes share schema_version, synthetic, status, stage, policy, rules,
    records, and matches. Records are enriched monotonically with triage/review.
    """
    keys(envelope, ("schema_version", "synthetic", "status", "stage", "policy",
                    "rules", "records", "matches"))
    require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1,
            "Unsupported schema version")
    require(envelope["synthetic"] is True, "Only synthetic fixtures are supported")
    require(envelope["status"] == "ok", "Invalid status")
    stage = envelope["stage"]
    require(stage in STAGES and (expected_stage is None or stage == expected_stage),
            "Unexpected pipeline stage")
    validate_policy(envelope["policy"])
    rules = envelope["rules"]
    keys(rules, ("categories", "fallback"))
    rule_map = {}
    for rule in sequence(rules["categories"], 30):
        keys(rule, ("name", "priority", "team", "owner", "required_evidence"))
        for field in ("name", "team", "owner"):
            token(rule[field], TAG)
        number(rule["priority"], 1, 4, integer=True)
        for item in sequence(rule["required_evidence"], 50):
            token(item, TAG)
        unique(rule["required_evidence"])
        require(rule["name"] not in rule_map, "Duplicate category")
        rule_map[rule["name"]] = rule
    require(rules["fallback"] in rule_map, "Unknown fallback category")
    ids = []
    record_map = {}
    for record in sequence(envelope["records"]):
        base = ("ticket_id", "account_id", "subscriber_ref", "residency", "signals",
                "urgent", "usage", "evidence", "billing_adjustment")
        extra = (() if stage == "intake" else ("triage",))
        if stage in ("review", "semantic"):
            extra += ("review",)
        keys(record, base + extra)
        for field in ("ticket_id", "account_id", "subscriber_ref"):
            token(record[field])
        ids.append(record["ticket_id"])
        record_map[record["ticket_id"]] = record
        check_region(record["residency"], envelope["policy"])
        keys(record["signals"], tuple(rule_map))
        for count in record["signals"].values():
            number(count, 0, 10000, integer=True)
        require(type(record["urgent"]) is bool, "Invalid urgency")
        keys(record["usage"], ("record_ids", "call_seconds", "data_mb"))
        for uid in sequence(record["usage"]["record_ids"]):
            token(uid)
        unique(record["usage"]["record_ids"])
        number(record["usage"]["call_seconds"])
        number(record["usage"]["data_mb"])
        require(isinstance(record["evidence"], dict), "Invalid evidence mapping")
        for requirement, source_ids in record["evidence"].items():
            token(requirement, TAG)
            require(bool(sequence(source_ids)), "Evidence needs a source")
            for source in source_ids:
                token(source)
            unique(source_ids)
        adjustment = record["billing_adjustment"]
        if adjustment is not None:
            keys(adjustment, ("amount", "reason_stated"))
            number(adjustment["amount"], -100000, 100000)
            require(adjustment["reason_stated"] is True,
                    "Billing adjustment requires a stated reason")
        if stage != "intake":
            result = record["triage"]
            keys(result, ("category", "priority", "team", "owner"))
            number(result["priority"], 1, 4, integer=True)
            winner = max(rule_map, key=lambda name: record["signals"][name])
            if record["signals"][winner] == 0:
                winner = rules["fallback"]
            rule = rule_map[winner]
            expected = {"category": winner, "priority": 1 if record["urgent"] else rule["priority"],
                        "team": rule["team"], "owner": rule["owner"]}
            require(result == expected, "Invalid triage handoff")
        if stage in ("review", "semantic"):
            review = record["review"]
            keys(review, ("checks", "gaps", "status", "certification_claim"))
            require(review["certification_claim"] is False, "Certification claim prohibited")
            for check in sequence(review["checks"], 51):
                keys(check, ("requirement", "source_ids", "satisfied"))
                require(type(check["satisfied"]) is bool, "Invalid evidence decision")
            required = rule_map[record["triage"]["category"]]["required_evidence"][:]
            if adjustment is not None and "adjustment_reason" not in required:
                required.append("adjustment_reason")
            checks = [{"requirement": item,
                       "source_ids": record["evidence"].get(item, []),
                       "satisfied": bool(record["evidence"].get(item))}
                      for item in required]
            gaps = [{"requirement": item["requirement"], "ticket_id": record["ticket_id"],
                     "owner": record["triage"]["owner"], "reason": "missing_evidence"}
                    for item in checks if not item["satisfied"]]
            require(review == {"checks": checks, "gaps": gaps,
                               "status": "gaps_found" if gaps else "evidence_present",
                               "certification_claim": False},
                    "Invalid review handoff")
    unique(ids)
    matches = sequence(envelope["matches"], 100)
    require(stage == "semantic" or not matches, "Premature search results")
    found = []
    for match in matches:
        keys(match, ("ticket_id", "residency", "score", "mode", "review_status", "gap_requirements"))
        require(match["ticket_id"] in record_map, "Unknown search result")
        record = record_map[match["ticket_id"]]
        require(match["residency"] == record["residency"], "Invalid search residency")
        require(match["review_status"] == record["review"]["status"]
                and match["gap_requirements"] ==
                [gap["requirement"] for gap in record["review"]["gaps"]],
                "Invalid search provenance")
        number(match["score"], 0, 1)
        require(match["mode"] in ("lexical", "injected_embedding"), "Invalid retrieval mode")
        found.append(match["ticket_id"])
    unique(found)
    require(matches == sorted(matches, key=lambda row: (-row["score"], row["ticket_id"])),
            "Search ranking is not deterministic")
    return envelope


def intake(payload):
    """Parse chat/CDR transport into the shared privacy-minimized envelope."""
    keys(payload, ("schema_version", "synthetic", "policy", "accounts", "tickets",
                   "cdr_csv", "documents", "query"), ("config",))
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1,
            "Unsupported schema version")
    require(payload["synthetic"] is True, "Only synthetic fixtures are supported")
    validate_policy(payload["policy"])
    policy = payload["policy"]
    config = validate_config(copy.deepcopy(payload.get("config", DEFAULT_CONFIG)))
    accounts = {}
    for account in sequence(payload["accounts"]):
        keys(account, ("id", "subscriber_ref", "residency", "consent", "phone", "imei"))
        token(account["id"])
        token(account["subscriber_ref"])
        check_region(account["residency"], policy)
        require(account["consent"] is True, "Subscriber permission required")
        require(isinstance(account["phone"], str)
                and re.fullmatch(r"\+999\d{9}", account["phone"]) is not None,
                "Phone must be an explicitly fictitious +999 fixture")
        require(isinstance(account["imei"], str)
                and re.fullmatch(r"000000000\d{6}", account["imei"]) is not None,
                "IMEI must be an explicitly fictitious fixture")
        require(account["id"] not in accounts, "Duplicate subscriber account")
        accounts[account["id"]] = account
    unique([account["subscriber_ref"] for account in accounts.values()])
    unique([account["phone"] for account in accounts.values()])
    unique([account["imei"] for account in accounts.values()])
    usage = {aid: {"record_ids": [], "call_seconds": 0, "data_mb": 0.0} for aid in accounts}
    require(isinstance(payload["cdr_csv"], str) and len(payload["cdr_csv"]) <= 2000000,
            "Invalid CSV input")
    reader = csv.DictReader(io.StringIO(payload["cdr_csv"]), strict=True)
    try:
        require(reader.fieldnames == list(CDR_FIELDS), "Invalid CSV header")
        seen = set()
        for index, row in enumerate(reader):
            require(index < 10000 and set(row) == set(CDR_FIELDS)
                    and all(isinstance(v, str) and v.strip() for v in row.values()),
                    "Invalid CSV row")
            token(row["id"])
            require(row["id"] not in seen, "Duplicate usage record")
            seen.add(row["id"])
            require(row["account_id"] in accounts, "Unknown usage account")
            account = accounts[row["account_id"]]
            check_region(row["residency"], policy, account["residency"])
            require(row["phone"] == account["phone"] and row["imei"] == account["imei"],
                    "Usage identity mismatch")
            require(re.fullmatch(r"\d+", row["call_seconds"]) is not None,
                    "Invalid call duration")
            try:
                seconds = int(row["call_seconds"])
                data_mb = float(row["data_mb"])
            except ValueError:
                raise ValidationError("Invalid usage volume") from None
            number(seconds, 0, 86400, integer=True)
            number(data_mb, 0, 1000000)
            total = usage[row["account_id"]]
            total["record_ids"].append(row["id"])
            total["call_seconds"] += seconds
            total["data_mb"] += data_mb
    except csv.Error:
        raise ValidationError("Malformed CSV") from None
    records = {}
    for ticket in sequence(payload["tickets"]):
        keys(ticket, ("id", "account_id", "residency", "subject", "transcript"),
             ("billing_adjustment",))
        token(ticket["id"])
        require(ticket["id"] not in records, "Duplicate network fault ticket")
        require(isinstance(ticket["account_id"], str) and ticket["account_id"] in accounts,
                "Unknown ticket account")
        account = accounts[ticket["account_id"]]
        check_region(ticket["residency"], policy, account["residency"])
        content = text(ticket["subject"]) + " " + text(ticket["transcript"])
        vocabulary = set(words(content))
        evidence = {}
        if usage[ticket["account_id"]]["record_ids"]:
            evidence["usage_record"] = usage[ticket["account_id"]]["record_ids"][:]
        adjustment = ticket.get("billing_adjustment")
        if adjustment is not None:
            keys(adjustment, ("amount", "reason"))
            number(adjustment["amount"], -100000, 100000)
            text(adjustment["reason"], "billing adjustment reason")
            evidence["adjustment_reason"] = [ticket["id"]]
            adjustment = {"amount": adjustment["amount"], "reason_stated": True}
        records[ticket["id"]] = {
            "ticket_id": ticket["id"], "account_id": ticket["account_id"],
            "subscriber_ref": account["subscriber_ref"], "residency": ticket["residency"],
            "signals": {rule["name"]: sum(keyword in vocabulary for keyword in rule["keywords"])
                        for rule in config["categories"]},
            "urgent": any(keyword in vocabulary for keyword in config["urgent_keywords"]),
            "usage": copy.deepcopy(usage[ticket["account_id"]]),
            "evidence": evidence, "billing_adjustment": adjustment,
        }
    document_ids = []
    # All provenance IDs occupy one namespace, preventing ambiguous source links.
    namespace = set(accounts) | set(records)
    require(len(namespace) == len(accounts) + len(records), "Ambiguous provenance identifier")
    require(not namespace.intersection(seen), "Ambiguous provenance identifier")
    namespace.update(seen)
    for document in sequence(payload["documents"]):
        keys(document, ("id", "ticket_id", "residency", "evidence"))
        token(document["id"])
        require(document["id"] not in namespace, "Ambiguous provenance identifier")
        namespace.add(document["id"])
        document_ids.append(document["id"])
        require(isinstance(document["ticket_id"], str) and document["ticket_id"] in records,
                "Unknown document ticket")
        record = records[document["ticket_id"]]
        check_region(document["residency"], policy, record["residency"])
        require(isinstance(document["evidence"], dict), "Invalid document evidence")
        for requirement, content in document["evidence"].items():
            token(requirement, TAG)
            require(isinstance(content, str) and len(content) <= 20000,
                    "Invalid evidence content")
            if content.strip():
                record["evidence"].setdefault(requirement, []).append(document["id"])
    unique(document_ids)
    query = validate_query(payload["query"], policy)
    envelope = {
        "schema_version": 1, "synthetic": True, "status": "ok", "stage": "intake",
        "policy": copy.deepcopy(policy),
        "rules": {"categories": [{k: copy.deepcopy(v) for k, v in rule.items() if k != "keywords"}
                                 for rule in config["categories"]],
                  "fallback": config["fallback"]},
        "records": list(records.values()), "matches": [],
    }
    return canonical_validate(envelope, "intake"), query


def triage(envelope):
    canonical_validate(envelope, "intake")
    result = copy.deepcopy(envelope)
    for record in result["records"]:
        rules = result["rules"]["categories"]
        chosen = max(rules, key=lambda rule: record["signals"][rule["name"]])
        if record["signals"][chosen["name"]] == 0:
            chosen = next(rule for rule in rules if rule["name"] == result["rules"]["fallback"])
        record["triage"] = {
            "category": chosen["name"],
            "priority": 1 if record["urgent"] else chosen["priority"],
            "team": chosen["team"], "owner": chosen["owner"],
        }
    result["stage"] = "triage"
    return canonical_validate(result, "triage")


def review(envelope):
    canonical_validate(envelope, "triage")
    result = copy.deepcopy(envelope)
    rules = {rule["name"]: rule for rule in result["rules"]["categories"]}
    for record in result["records"]:
        required = rules[record["triage"]["category"]]["required_evidence"][:]
        if record["billing_adjustment"] is not None and "adjustment_reason" not in required:
            required.append("adjustment_reason")
        checks = [{"requirement": item, "source_ids": record["evidence"].get(item, [])[:],
                   "satisfied": bool(record["evidence"].get(item))} for item in required]
        gaps = [{"requirement": check["requirement"], "ticket_id": record["ticket_id"],
                 "owner": record["triage"]["owner"], "reason": "missing_evidence"}
                for check in checks if not check["satisfied"]]
        record["review"] = {"checks": checks, "gaps": gaps,
                            "status": "gaps_found" if gaps else "evidence_present",
                            "certification_claim": False}
    result["stage"] = "review"
    return canonical_validate(result, "review")


def validate_query(query, policy):
    keys(query, ("text", "residency", "limit"))
    text(query["text"], "search query", 1000)
    require(bool(words(query["text"])), "Search query needs alphabetic terms")
    check_region(query["residency"], policy)
    number(query["limit"], 1, 100, integer=True)
    return copy.deepcopy(query)


def semantic_terms(value):
    return [SYNONYMS.get(word, word) for word in words(value)]


def embedding_vector(callback, content):
    try:
        vector = callback(content)
    except Exception:
        raise ValidationError("Injected embedding callable failed") from None
    require(isinstance(vector, (list, tuple)) and 0 < len(vector) <= 4096,
            "Invalid embedding vector")
    for value in vector:
        number(value, -1e100, 1e100)
    require(any(value != 0 for value in vector), "Zero embedding vector")
    norm = math.hypot(*vector)
    return [value / norm for value in vector]


def semantic_search(envelope, query, embedder=None):
    """Build a residency-partitioned in-memory index over validated reviews.

    Default ranking is cosine TF-IDF with a fixed telecom synonym vocabulary.
    An explicitly injected callable may replace vectors; it receives only
    minimized document tags and normalized query terms, never source records.
    """
    canonical_validate(envelope, "review")
    query = validate_query(query, envelope["policy"])
    require(embedder is None or callable(embedder), "Embedding interface must be callable")
    result = copy.deepcopy(envelope)
    candidates = [record for record in result["records"]
                  if record["residency"] == query["residency"]]
    documents = []
    for record in candidates:
        tags = [record["triage"]["category"], record["triage"]["team"],
                record["review"]["status"]]
        tags.extend(check["requirement"] for check in record["review"]["checks"])
        tags.extend("gap " + gap["requirement"] for gap in record["review"]["gaps"])
        documents.append(" ".join(semantic_terms(" ".join(tags))))
    query_terms = semantic_terms(query["text"])
    scores = []
    if embedder is not None and candidates:
        query_vector = embedding_vector(embedder, " ".join(query_terms))
        for document in documents:
            vector = embedding_vector(embedder, document)
            require(len(vector) == len(query_vector), "Embedding dimensions differ")
            scores.append(max(0.0, min(1.0, sum(a * b for a, b in zip(vector, query_vector)))))
    elif candidates:
        counters = [Counter(document.split()) for document in documents]
        qcounter = Counter(query_terms)
        frequencies = Counter(term for counter in counters for term in counter)
        idf = {term: math.log((1 + len(counters)) / (1 + frequencies[term])) + 1
               for term in set(frequencies) | set(qcounter)}
        qvector = {term: count * idf[term] for term, count in qcounter.items()}
        qnorm = math.sqrt(sum(value * value for value in qvector.values()))
        for counter in counters:
            vector = {term: count * idf[term] for term, count in counter.items()}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            score = sum(value * qvector.get(term, 0) for term, value in vector.items())
            scores.append(min(1.0, score / (norm * qnorm)) if norm and qnorm else 0.0)
    matches = []
    for record, score in zip(candidates, scores):
        if score > 0:
            matches.append({
                "ticket_id": record["ticket_id"], "residency": record["residency"],
                "score": round(score, 8),
                "mode": "injected_embedding" if embedder is not None else "lexical",
                "review_status": record["review"]["status"],
                "gap_requirements": [gap["requirement"] for gap in record["review"]["gaps"]],
            })
    result["matches"] = sorted(matches, key=lambda row: (-row["score"], row["ticket_id"]))[:query["limit"]]
    result["stage"] = "semantic"
    return canonical_validate(result, "semantic")


def run_pipeline(payload, embedder=None):
    envelope, query = intake(payload)
    return semantic_search(review(triage(envelope)), query, embedder=embedder)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py input.json")
        with open(argv[0], encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=reject_duplicate_keys)
        result = run_pipeline(payload)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Never echo raw values, file paths, or provider exception messages.
        print(json.dumps({"status": "error", "error": "Input, file, or pipeline validation failed"}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
