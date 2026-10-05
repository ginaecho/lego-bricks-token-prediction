"""Offline, synthetic insurance reference pipeline; no compliance certification.

Run: python -B implementation.py example_input.json
Optional retrieval and embedding callables are injected into run_pipeline().
ACORD-style formats below are illustrative, not certified ACORD schemas.
"""

import copy
import hashlib
import json
import math
import re
import sys
from datetime import date, datetime
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


VERSION = "1.0"
STAGES = ("research", "guided", "behavior", "semantic")
FACTORS = {"coverage_amount", "property_type", "construction_year",
           "prior_claim_count", "vehicle_age"}
STEPS = ("review_minimization", "review_disclosures", "choose_preferences")
KINDS = {"Policy": "policy", "Claim": "claim",
         "UnderwritingSubmission": "underwriting_submission"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, required, optional=(), label="object"):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(required) <= set(value), f"{label} missing required fields")
    require(set(value) <= set(required) | set(optional), f"{label} has unknown fields")


def text(value, label, limit=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
            f"{label} must be nonempty bounded text")
    return value.strip()


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Z][A-Z0-9_-]{0,39}", value) is not None,
            "invalid synthetic identifier")
    return value


def number(value, label, minimum=0, maximum=1e12):
    require(type(value) in (int, float) and minimum <= value <= maximum
            and math.isfinite(value), f"invalid {label}")
    return value


def timestamp(value):
    text(value, "timestamp", 50)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError("invalid timestamp") from None
    require(result.tzinfo is not None, "timestamp needs timezone")
    return result


def calendar_date(value):
    require(isinstance(value, str) and
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None, "invalid date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValidationError("invalid date") from None


def strings(value, label, allowed=None, nonempty=False):
    require(isinstance(value, list), f"{label} must be an array")
    require(not nonempty or len(value) > 0, f"{label} must not be empty")
    for item in value:
        text(item, label, 100)
        if allowed is not None:
            require(item in allowed, f"unsupported {label}")
    require(len(set(value)) == len(value), f"duplicate {label}")
    return value


def safe_url(value, hosts):
    text(value, "URL", 1000)
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.hostname in hosts
                 and parsed.netloc == parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and re.fullmatch(r"/[A-Za-z0-9/_\-.]*", parsed.path) is not None)
    except ValueError:
        valid = False
    require(valid, "URL must use an exact allowlisted HTTPS host without credentials or query")


def validate_input(data):
    fields(data, ("schema_version", "synthetic", "as_of", "research", "guided",
                  "behavior", "semantic"), label="input")
    require(data["schema_version"] == VERSION, "unsupported schema version")
    require(data["synthetic"] is True, "only explicitly synthetic fixtures are accepted")
    now = timestamp(data["as_of"])
    research = data["research"]
    fields(research, ("allowlisted_hosts", "documents"), label="research")
    hosts = strings(research["allowlisted_hosts"], "hosts", nonempty=True)
    for host in hosts:
        require(re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}", host)
                is not None, "invalid allowlisted host")
    require(isinstance(research["documents"], list) and
            0 < len(research["documents"]) <= 100, "documents must contain 1..100 items")
    urls = set()
    for document in research["documents"]:
        fields(document, ("url", "format", "content"), label="document")
        safe_url(document["url"], hosts)
        require(document["url"] not in urls, "duplicate source URL")
        urls.add(document["url"])
        require(document["format"] in ("acord-json", "acord-xml", "claim-text"),
                "unsupported source format")
        validate_content(document["content"], document["format"])
    guided = data["guided"]
    fields(guided, ("completed_steps", "acknowledge_minimization",
                    "acknowledged_factors", "preferred_perils"), label="guided")
    steps = strings(guided["completed_steps"], "steps", STEPS)
    require(steps == list(STEPS[:len(steps)]), "onboarding prerequisites not satisfied")
    require(type(guided["acknowledge_minimization"]) is bool, "invalid minimization acknowledgment")
    strings(guided["acknowledged_factors"], "acknowledged factors", FACTORS)
    strings(guided["preferred_perils"], "preferred perils")
    behavior = data["behavior"]
    fields(behavior, ("events", "half_life_days"), label="behavior")
    number(behavior["half_life_days"], "half life", 0.01, 36500)
    require(isinstance(behavior["events"], list) and len(behavior["events"]) <= 10000,
            "events must be a bounded array")
    for event in behavior["events"]:
        fields(event, ("entity_id", "action", "timestamp"), label="event")
        identifier(event["entity_id"])
        require(event["action"] in ("browse", "purchase"), "invalid event action")
        require(timestamp(event["timestamp"]) <= now, "future event")
    semantic = data["semantic"]
    fields(semantic, ("query", "top_k"), label="semantic")
    text(semantic["query"], "query", 500)
    require(type(semantic["top_k"]) is int and 1 <= semantic["top_k"] <= 100,
            "top_k must be an integer in 1..100")


def validate_content(content, format_name):
    if format_name == "acord-json":
        require(isinstance(content, dict), "ACORD JSON content must be an object")
    else:
        text(content, "source content", 200000)
    try:
        encoded = json.dumps(content, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        raise ValidationError("source content must be finite JSON data") from None
    require(len(encoded) <= 200000, "source content too large")


def xml_records(content):
    require("<!DOCTYPE" not in content.upper() and "<!ENTITY" not in content.upper(),
            "XML DTDs and entities are forbidden")
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        raise ValidationError("malformed ACORD XML") from None
    require(root.tag == "ACORD" and not root.attrib, "expected plain ACORD XML root")
    result = {}
    for child in root:
        require(child.tag in KINDS and not child.attrib, "unknown ACORD XML entity")
        record = {}
        for field in child:
            require(field.tag not in record and not field.attrib, "duplicate or attributed XML field")
            if field.tag in ("CoveredPerils", "DisclosedFactors"):
                item_tag = "Peril" if field.tag == "CoveredPerils" else "Factor"
                require(all(x.tag == item_tag and not x.attrib and len(x) == 0 for x in field),
                        "invalid XML list")
                value = [x.text or "" for x in field]
            elif field.tag in ("Policyholder", "UnderwritingFactors"):
                value = {}
                for item in field:
                    require(item.tag not in value and not item.attrib and len(item) == 0,
                            "invalid XML object")
                    value[item.tag] = item.text or ""
            else:
                require(len(field) == 0, "unexpected nested XML")
                value = field.text or ""
                if field.tag in ("CoverageLimit", "LossAmount"):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValidationError("invalid XML amount") from None
            record[field.tag] = value
        result.setdefault(child.tag, []).append(record)
    return {"ACORD": result}


def claim_text_records(content):
    mapping = {"Claim-ID": "Id", "Policy-ID": "PolicyId", "Loss-Amount": "LossAmount",
               "Loss-Date": "LossDate", "Peril": "Peril"}
    record = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        require(separator and key in mapping, "unknown claim-form field")
        require(mapping[key] not in record, "duplicate claim-form field")
        record[mapping[key]] = value.strip()
    if "LossAmount" in record:
        try:
            record["LossAmount"] = float(record["LossAmount"])
        except ValueError:
            raise ValidationError("invalid claim-form loss amount") from None
    return {"ACORD": {"Claim": [record]}}


def minimize_holder(holder):
    fields(holder, ("Ref",), ("Name", "Email", "PropertyAddress", "VIN"), "policyholder")
    for key, value in holder.items():
        text(value, "synthetic policyholder field", 200)
    return identifier(holder["Ref"])


def normalize_record(kind, raw, provenance):
    common = {"entity_type": KINDS[kind], "provenance": provenance}
    if kind == "Policy":
        fields(raw, ("Id", "Title", "Description", "Policyholder", "CoverageLimit",
                     "CoveredPerils", "EffectiveDate", "ExpiryDate"), label="policy")
        start, end = calendar_date(raw["EffectiveDate"]), calendar_date(raw["ExpiryDate"])
        require(start <= end, "invalid policy coverage period")
        common.update(id=identifier(raw["Id"]), title=text(raw["Title"], "title", 200),
                      description=text(raw["Description"], "description"),
                      holder_ref=minimize_holder(raw["Policyholder"]),
                      coverage_limit=number(raw["CoverageLimit"], "coverage limit", 0.01),
                      covered_perils=strings(raw["CoveredPerils"], "covered perils", nonempty=True),
                      effective_date=raw["EffectiveDate"], expiry_date=raw["ExpiryDate"])
    elif kind == "Claim":
        fields(raw, ("Id", "PolicyId", "LossAmount", "LossDate", "Peril"), label="claim")
        calendar_date(raw["LossDate"])
        common.update(id=identifier(raw["Id"]), policy_id=identifier(raw["PolicyId"]),
                      loss_amount=number(raw["LossAmount"], "loss amount", 0.01),
                      loss_date=raw["LossDate"], peril=text(raw["Peril"], "peril", 100))
    else:
        fields(raw, ("Id", "PolicyId", "Policyholder", "UnderwritingFactors",
                     "DisclosedFactors"), label="underwriting submission")
        factors = raw["UnderwritingFactors"]
        require(isinstance(factors, dict), "underwriting factors must be an object")
        disclosures = strings(raw["DisclosedFactors"], "disclosed factors", FACTORS)
        require(set(factors) == set(disclosures), "all underwriting factors must be disclosed")
        require(bool(factors), "underwriting factors cannot be empty")
        for key, value in factors.items():
            require(key in FACTORS, "unsupported underwriting factor")
            if type(value) in (int, float):
                number(value, "underwriting factor")
            else:
                text(value, "underwriting factor", 100)
        common.update(id=identifier(raw["Id"]), policy_id=identifier(raw["PolicyId"]),
                      holder_ref=minimize_holder(raw["Policyholder"]),
                      underwriting_factors=copy.deepcopy(factors),
                      disclosed_factors=sorted(disclosures))
    return common


def assess_claim(claim, policy):
    """Uniform coverage triage, never automated approval, denial, or settlement."""
    reasons = []
    if claim["peril"] not in policy["covered_perils"]:
        reasons.append("Reported peril is not listed in policy coverage.")
    if claim["loss_amount"] > policy["coverage_limit"]:
        reasons.append("Reported loss exceeds the stated policy limit.")
    if not policy["effective_date"] <= claim["loss_date"] <= policy["expiry_date"]:
        reasons.append("Reported loss date is outside the stated coverage period.")
    return {"decision": "manual_review" if reasons else "eligible_for_review",
            "reasons": reasons or ["Reported peril, amount, and date meet stated coverage criteria."],
            "human_review_required": True,
            "rule_inputs": ["covered_perils", "coverage_limit", "coverage_period",
                            "peril", "loss_amount", "loss_date"]}


def validate_snapshot(state, expected_stage):
    """The shared envelope and industry rules are checked at every handoff."""
    fields(state, ("schema_version", "synthetic", "status", "stage", "as_of",
                   "sources", "findings", "guided", "behavior", "semantic"), label="snapshot")
    require(state["schema_version"] == VERSION and state["synthetic"] is True
            and state["status"] == "ok" and state["stage"] == expected_stage,
            "invalid snapshot metadata or handoff order")
    timestamp(state["as_of"])
    require(isinstance(state["sources"], list) and bool(state["sources"]), "missing provenance")
    source_map = {}
    for source in state["sources"]:
        fields(source, ("url", "format", "sha256", "retrieved_at", "record_ids"), label="source")
        require(source["url"] not in source_map, "duplicate provenance")
        require(isinstance(source["sha256"], str) and
                re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, "invalid digest")
        require(source["retrieved_at"] == state["as_of"], "invalid retrieval timestamp")
        require(source["format"] in ("acord-json", "acord-xml", "claim-text"), "invalid format")
        strings(source["record_ids"], "record IDs", nonempty=True)
        source_map[source["url"]] = source
    require(isinstance(state["findings"], list) and bool(state["findings"]), "empty findings")
    records = {}
    for record in state["findings"]:
        require(isinstance(record, dict), "finding must be an object")
        kind = record.get("entity_type")
        base = ("id", "entity_type", "provenance")
        if kind == "policy":
            keys = ("title", "description", "holder_ref", "coverage_limit", "covered_perils",
                    "effective_date", "expiry_date")
        elif kind == "claim":
            keys = ("policy_id", "loss_amount", "loss_date", "peril", "assessment")
        else:
            require(kind == "underwriting_submission", "invalid entity type")
            keys = ("policy_id", "holder_ref", "underwriting_factors", "disclosed_factors")
        fields(record, base + keys, label="finding")
        identifier(record["id"])
        require(record["id"] not in records, "duplicate entity ID")
        records[record["id"]] = record
        provenance = record["provenance"]
        fields(provenance, ("url", "pointer", "sha256"), label="provenance")
        source = source_map.get(provenance["url"])
        require(source is not None and record["id"] in source["record_ids"]
                and source["sha256"] == provenance["sha256"], "broken provenance")
        text(provenance["pointer"], "source pointer", 100)
        if "holder_ref" in record:
            identifier(record["holder_ref"])
        if kind == "policy":
            text(record["title"], "title", 200)
            text(record["description"], "description")
            number(record["coverage_limit"], "coverage limit", 0.01)
            strings(record["covered_perils"], "covered perils", nonempty=True)
            require(calendar_date(record["effective_date"]) <= calendar_date(record["expiry_date"]),
                    "invalid coverage period")
        elif kind == "claim":
            number(record["loss_amount"], "loss amount", 0.01)
            require(calendar_date(record["loss_date"]) <= timestamp(state["as_of"]).date(),
                    "future loss date")
            text(record["peril"], "peril", 100)
        else:
            strings(record["disclosed_factors"], "disclosed factors", FACTORS, True)
            require(isinstance(record["underwriting_factors"], dict) and
                    set(record["disclosed_factors"]) == set(record["underwriting_factors"]),
                    "hidden underwriting factors")
            for value in record["underwriting_factors"].values():
                if type(value) in (int, float):
                    number(value, "underwriting factor")
                else:
                    text(value, "underwriting factor", 100)
    policies = {key: value for key, value in records.items() if value["entity_type"] == "policy"}
    require(bool(policies), "research requires at least one policy")
    for record in records.values():
        if record["entity_type"] != "policy":
            require(record["policy_id"] in policies, "unknown linked policy")
            policy = policies[record["policy_id"]]
            if record["entity_type"] == "claim":
                require(record["assessment"] == assess_claim(record, policy),
                        "claim decisions must use uniform stated reasons and human review")
            else:
                require(record["holder_ref"] == policy["holder_ref"], "submission holder mismatch")
    for source in state["sources"]:
        actual = [r["id"] for r in records.values() if r["provenance"]["url"] == source["url"]]
        require(actual == source["record_ids"], "source membership mismatch")
    index = STAGES.index(expected_stage)
    for position, key in enumerate(("guided", "behavior", "semantic"), start=1):
        require((state[key] is not None) == (index >= position), "invalid stage presence")
    if index >= 1:
        guided = state["guided"]
        fields(guided, ("completed_steps", "next_step", "ready", "progress",
                        "preferred_perils", "disclosed_factors", "eligible_policy_ids",
                        "minimized_fields"), label="guided output")
        steps = guided["completed_steps"]
        strings(steps, "steps", STEPS)
        require(steps == list(STEPS[:len(steps)]), "invalid prerequisite progress")
        require(type(guided["ready"]) is bool and guided["ready"] == (len(steps) == len(STEPS)),
                "invalid readiness")
        require(guided["progress"] == len(steps) / len(STEPS), "invalid progress")
        require(guided["next_step"] == (STEPS[len(steps)] if len(steps) < len(STEPS) else None),
                "invalid next step")
        require(guided["eligible_policy_ids"] == sorted(policies), "policy handoff mismatch")
        disclosed = sorted({f for r in records.values()
                            for f in r.get("disclosed_factors", [])})
        require(guided["disclosed_factors"] == disclosed, "disclosure handoff mismatch")
        strings(guided["preferred_perils"], "preferred perils",
                {p for policy in policies.values() for p in policy["covered_perils"]})
        require(guided["minimized_fields"] == ["Name", "Email", "PropertyAddress", "VIN"],
                "invalid minimization declaration")
    if index >= 2:
        behavior = state["behavior"]
        fields(behavior, ("mode", "rankings", "half_life_days", "events_used"), label="behavior output")
        require(behavior["mode"] in ("blocked", "cold_start", "personalized"), "invalid behavior mode")
        number(behavior["half_life_days"], "half life", 0.01, 36500)
        require(type(behavior["events_used"]) is int and behavior["events_used"] >= 0,
                "invalid event count")
        require(isinstance(behavior["rankings"], list), "invalid rankings")
        ids = []
        for ranking in behavior["rankings"]:
            fields(ranking, ("policy_id", "score", "reasons"), label="ranking")
            require(ranking["policy_id"] in policies, "unknown ranked policy")
            number(ranking["score"], "behavior score")
            strings(ranking["reasons"], "ranking reasons", nonempty=True)
            ids.append(ranking["policy_id"])
        expected_ids = sorted(policies) if state["guided"]["ready"] else []
        require(sorted(ids) == expected_ids, "rankings must cover eligible policies once")
        require((behavior["mode"] == "blocked") == (not state["guided"]["ready"]),
                "onboarding gate bypass")
        require(behavior["rankings"] == sorted(behavior["rankings"],
                key=lambda r: (-r["score"], r["policy_id"])), "invalid ranking order")
    if index >= 3:
        semantic = state["semantic"]
        fields(semantic, ("query", "mode", "index", "results"), label="search output")
        text(semantic["query"], "query", 500)
        require(semantic["mode"] in ("blocked", "lexical", "injected_embedding"), "invalid search mode")
        require((semantic["mode"] == "blocked") == (not state["guided"]["ready"]),
                "search onboarding gate bypass")
        require(isinstance(semantic["index"], list) and isinstance(semantic["results"], list),
                "invalid search arrays")
        expected = sorted(records) if state["guided"]["ready"] else []
        require([d.get("entity_id") for d in semantic["index"]] == expected, "invalid index handoff")
        for document in semantic["index"]:
            fields(document, ("entity_id", "terms"), label="indexed document")
            strings(document["terms"], "index terms")
        seen = set()
        for result in semantic["results"]:
            fields(result, ("entity_id", "entity_type", "policy_id", "score", "relevance",
                            "behavior_score", "provenance"), label="search result")
            key = result["entity_id"]
            require(key in records and key not in seen, "invalid search entity")
            seen.add(key)
            original = records[key]
            require(result["entity_type"] == original["entity_type"] and
                    result["provenance"] == original["provenance"] and
                    result["policy_id"] == original.get("policy_id", key), "search provenance mismatch")
            number(result["score"], "search score", 0, 2)
            number(result["relevance"], "relevance", 0, 1.000000001)
            number(result["behavior_score"], "behavior score", 0, 1)
        require(semantic["results"] == sorted(semantic["results"],
                key=lambda r: (-r["score"], r["entity_id"])), "invalid search order")
        require(semantic["mode"] != "blocked" or not semantic["results"], "blocked search has results")
    return state


def research_stage(data, retriever=None):
    validate_input(data)
    findings, sources = [], []
    for document in data["research"]["documents"]:
        try:
            content = (retriever(document["url"]) if retriever else
                       copy.deepcopy(document["content"]))
        except Exception:
            raise ValidationError("injected retrieval failed") from None
        validate_content(content, document["format"])
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if document["format"] == "acord-xml":
            parsed = xml_records(content)
        elif document["format"] == "claim-text":
            parsed = claim_text_records(content)
        else:
            parsed = content
        fields(parsed, ("ACORD",), label="ACORD document")
        entities = parsed["ACORD"]
        fields(entities, (), KINDS, label="ACORD entities")
        require(bool(entities), "empty ACORD document")
        ids = []
        for kind in sorted(entities):
            require(isinstance(entities[kind], list) and bool(entities[kind]),
                    "ACORD entity collections must be nonempty arrays")
            for index, raw in enumerate(entities[kind]):
                provenance = {"url": document["url"], "pointer": f"/ACORD/{kind}/{index}",
                              "sha256": digest}
                record = normalize_record(kind, raw, provenance)
                findings.append(record)
                ids.append(record["id"])
        sources.append({"url": document["url"], "format": document["format"],
                        "sha256": digest, "retrieved_at": data["as_of"], "record_ids": ids})
    policies = {r["id"]: r for r in findings if r["entity_type"] == "policy"}
    for record in findings:
        if record["entity_type"] == "claim":
            require(record["policy_id"] in policies, "claim references unknown policy")
            record["assessment"] = assess_claim(record, policies[record["policy_id"]])
    return validate_snapshot({"schema_version": VERSION, "synthetic": True, "status": "ok",
                              "stage": "research", "as_of": data["as_of"],
                              "sources": sources, "findings": findings,
                              "guided": None, "behavior": None, "semantic": None}, "research")


def guided_stage(previous, config):
    validate_snapshot(previous, "research")
    state = copy.deepcopy(previous)
    steps = config["completed_steps"]
    strings(steps, "steps", STEPS)
    require(steps == list(STEPS[:len(steps)]), "onboarding prerequisites not satisfied")
    factors = sorted({factor for record in state["findings"]
                      for factor in record.get("disclosed_factors", [])})
    if len(steps) >= 1:
        require(config["acknowledge_minimization"] is True, "minimization acknowledgment required")
    if len(steps) >= 2:
        require(sorted(config["acknowledged_factors"]) == factors,
                "all retrieved underwriting factors must be acknowledged")
    policies = [r for r in state["findings"] if r["entity_type"] == "policy"]
    strings(config["preferred_perils"], "preferred perils",
            {peril for p in policies for peril in p["covered_perils"]})
    state["guided"] = {"completed_steps": list(steps),
                       "next_step": STEPS[len(steps)] if len(steps) < len(STEPS) else None,
                       "ready": len(steps) == len(STEPS), "progress": len(steps) / len(STEPS),
                       "preferred_perils": list(config["preferred_perils"]),
                       "disclosed_factors": factors,
                       "eligible_policy_ids": sorted(p["id"] for p in policies),
                       "minimized_fields": ["Name", "Email", "PropertyAddress", "VIN"]}
    state["stage"] = "guided"
    return validate_snapshot(state, "guided")


def behavior_stage(previous, config):
    validate_snapshot(previous, "guided")
    state = copy.deepcopy(previous)
    records = {r["id"]: r for r in state["findings"]}
    for event in config["events"]:
        require(event["entity_id"] in records, "event references unknown entity")
    rankings = []
    ready = state["guided"]["ready"]
    mode = "blocked" if not ready else ("personalized" if config["events"] else "cold_start")
    if ready:
        now = timestamp(state["as_of"])
        scores = {key: 0.0 for key in state["guided"]["eligible_policy_ids"]}
        counts = {key: 0 for key in scores}
        for event in config["events"]:
            record = records[event["entity_id"]]
            policy_id = record.get("policy_id", record["id"])
            age_days = (now - timestamp(event["timestamp"])).total_seconds() / 86400
            require(age_days >= 0, "future event")
            scores[policy_id] += (3 if event["action"] == "purchase" else 1) * (
                2 ** (-age_days / config["half_life_days"]))
            counts[policy_id] += 1
        for policy_id, score in scores.items():
            policy = records[policy_id]
            affinity = bool(set(policy["covered_perils"]) & set(state["guided"]["preferred_perils"]))
            score += 0.25 if affinity else 0
            reasons = ["Recency-weighted activity." if counts[policy_id]
                       else "No observed activity; deterministic discovery fallback."]
            if affinity:
                reasons.append("Matches explicitly selected coverage preferences.")
            rankings.append({"policy_id": policy_id, "score": round(score, 10), "reasons": reasons})
        rankings.sort(key=lambda item: (-item["score"], item["policy_id"]))
    state["behavior"] = {"mode": mode, "rankings": rankings,
                         "half_life_days": config["half_life_days"],
                         "events_used": len(config["events"]) if ready else 0}
    state["stage"] = "behavior"
    return validate_snapshot(state, "behavior")


def tokens(value):
    aliases = {"automobile": "auto", "car": "auto", "vehicle": "auto",
               "house": "home", "residence": "home", "flooding": "flood"}
    return [aliases.get(word, word) for word in re.findall(r"[a-z0-9]+", value.lower())]


def searchable(record):
    parts = [record["id"], record["entity_type"].replace("_", " ")]
    if record["entity_type"] == "policy":
        parts += [record["title"], record["description"], " ".join(record["covered_perils"])]
    elif record["entity_type"] == "claim":
        parts += [record["peril"], record["assessment"]["decision"].replace("_", " ")]
    else:
        parts += [factor.replace("_", " ") for factor in record["disclosed_factors"]]
    return " ".join(parts)


def embedding(callable_, value, dimension=None):
    try:
        vector = callable_(value)
    except Exception:
        raise ValidationError("injected embedding failed") from None
    require(isinstance(vector, (list, tuple)) and 0 < len(vector) <= 2048,
            "embedding must be a bounded nonempty vector")
    require(dimension is None or len(vector) == dimension, "embedding dimension mismatch")
    for component in vector:
        number(component, "embedding component", -1e6, 1e6)
    norm = math.sqrt(sum(component * component for component in vector))
    require(norm > 0, "zero embedding vector")
    return [component / norm for component in vector]


def semantic_stage(previous, config, embedder=None):
    validate_snapshot(previous, "behavior")
    state = copy.deepcopy(previous)
    query = text(config["query"], "query", 500)
    query_tokens = tokens(query)
    require(bool(query_tokens), "query must contain searchable terms")
    documents, results = [], []
    mode = "blocked"
    if state["guided"]["ready"]:
        mode = "injected_embedding" if embedder else "lexical"
        records = sorted(state["findings"], key=lambda r: r["id"])
        texts = [searchable(r) for r in records]
        term_lists = [tokens(value) for value in texts]
        vocabulary = sorted(set(query_tokens).union(*(set(t) for t in term_lists)))
        idf = {term: math.log((1 + len(records)) /
               (1 + sum(term in terms for terms in term_lists))) + 1 for term in vocabulary}

        def lexical_vector(terms):
            vector = [terms.count(term) * idf[term] for term in vocabulary]
            norm = math.sqrt(sum(x * x for x in vector))
            return [x / norm for x in vector] if norm else vector

        query_vector = lexical_vector(query_tokens)
        query_embedding = embedding(embedder, query) if embedder else None
        scores = {r["policy_id"]: r["score"] for r in state["behavior"]["rankings"]}
        maximum = max(scores.values(), default=0) or 1
        for record, doc_text, terms in zip(records, texts, term_lists):
            documents.append({"entity_id": record["id"], "terms": sorted(set(terms))})
            relevance = sum(a * b for a, b in zip(query_vector, lexical_vector(terms)))
            if embedder:
                vector = embedding(embedder, doc_text, len(query_embedding))
                cosine = max(0.0, min(1.0, sum(a * b for a, b in zip(query_embedding, vector))))
                relevance = 0.5 * relevance + 0.5 * cosine
            relevance = min(1.0, relevance)
            if relevance <= 0:
                continue
            policy_id = record.get("policy_id", record["id"])
            prior = scores[policy_id] / maximum
            results.append({"entity_id": record["id"], "entity_type": record["entity_type"],
                            "policy_id": policy_id, "score": round(relevance + 0.1 * prior, 10),
                            "relevance": round(relevance, 10), "behavior_score": round(prior, 10),
                            "provenance": copy.deepcopy(record["provenance"])})
        results.sort(key=lambda item: (-item["score"], item["entity_id"]))
        results = results[:config["top_k"]]
    state["semantic"] = {"query": query, "mode": mode, "index": documents, "results": results}
    state["stage"] = "semantic"
    return validate_snapshot(state, "semantic")


def run_pipeline(data, retriever=None, embedder=None):
    validate_input(data)
    state = research_stage(data, retriever)
    state = guided_stage(state, data["guided"])
    state = behavior_stage(state, data["behavior"])
    return semantic_stage(state, data["semantic"], embedder)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py example_input.json")
        with open(argv[0], "r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 ValidationError("nonfinite JSON constant")))
        result = run_pipeline(data)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        message = str(error) if isinstance(error, ValidationError) else "Cannot read valid bounded JSON input."
        print(json.dumps({"schema_version": VERSION, "status": "error", "message": message}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
