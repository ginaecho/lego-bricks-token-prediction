"""Synthetic, deterministic telecommunications support -> onboarding -> discovery.

Run: python -B implementation.py example_input.json
All processing is local. Demonstrative privacy rules are not legal certification.
"""

import copy
import csv
import datetime
import io
import json
import math
import re
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names.split()), label + ": invalid fields")


def text(value, label, limit=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit, label + ": invalid text")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label + ": invalid number")


def resident(record, region):
    require(record.get("residency") == region, "record residency mismatch")


def identifier(value, prefix):
    require(isinstance(value, str) and re.fullmatch(prefix + r"_[a-z0-9]{1,24}", value) is not None,
            "invalid " + prefix + " identifier")


def usage_records(source, subscriber, region):
    text(source, "usage CSV", 100000)
    reader = csv.DictReader(io.StringIO(source), strict=True)
    records, seen = [], set()
    try:
        require(reader.fieldnames == ["record_id", "subscriber_id", "residency", "timestamp", "call_minutes", "data_mb"],
                "invalid usage CSV header")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "invalid CSV row")
            identifier(row["record_id"], "cdr")
            require(row["record_id"] not in seen, "duplicate usage record")
            seen.add(row["record_id"])
            resident(row, region)
            require(row["subscriber_id"] == subscriber, "usage subscriber mismatch")
            try:
                stamp = datetime.datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                require(stamp.tzinfo is not None, "usage timestamp needs timezone")
                minutes, data = float(row["call_minutes"]), float(row["data_mb"])
            except (TypeError, ValueError) as exc:
                raise ValidationError("invalid CSV date or volume") from exc
            number(minutes, "call minutes")
            number(data, "data MB")
            records.append((minutes, data))
    except csv.Error as exc:
        raise ValidationError("malformed usage CSV") from exc
    total_minutes = sum(r[0] for r in records)
    total_data = sum(r[1] for r in records)
    number(total_minutes, "total call minutes")
    number(total_data, "total data MB")
    return {"residency": region, "record_count": len(records),
            "call_minutes": round(total_minutes, 2), "data_gb": round(total_data / 1024, 3)}


def validate(value, phase="input"):
    """One validation boundary for external input and every pipeline handoff."""
    require(isinstance(value, dict), "document must be an object")
    require(value.get("schema_version") == "1.0" and value.get("synthetic") is True,
            "only schema 1.0 clearly labeled synthetic data is accepted")
    require(value.get("residency") in ("EU", "US"), "unsupported residency")
    region = value["residency"]
    if phase != "input":
        validate_state(value, phase)
        return value
    fields(value, "schema_version synthetic residency subscriber tickets usage_csv chat catalog", "input")
    account = value["subscriber"]
    fields(account, "subscriber_id residency phone imei processing_consent discovery_consent current_plan goal completed_steps",
           "subscriber")
    resident(account, region)
    identifier(account["subscriber_id"], "sub")
    require(isinstance(account["phone"], str) and re.fullmatch(r"\+1999555\d{4}", account["phone"]) is not None,
            "use fictitious +1-999-555 phone numbers")
    require(isinstance(account["imei"], str) and re.fullmatch(r"00000000\d{7}", account["imei"]) is not None,
            "use fictitious zero-prefix IMEIs")
    require(account["processing_consent"] is True, "subscriber processing consent required")
    require(type(account["discovery_consent"]) is bool, "invalid discovery consent")
    identifier(account["current_plan"], "plan")
    require(account["goal"] in ("activate", "reduce_cost", "more_data"), "unsupported goal")
    require(isinstance(account["completed_steps"], list) and
            all(s in ("activate_sim", "check_connection", "review_usage") for s in account["completed_steps"]) and
            len(set(account["completed_steps"])) == len(account["completed_steps"]), "invalid completed steps")
    for key in ("tickets", "chat", "catalog"):
        require(isinstance(value[key], list) and len(value[key]) <= 100, key + " must be a bounded list")
    seen = set()
    for ticket in value["tickets"]:
        fields(ticket, "ticket_id subscriber_id residency category status billing_adjustment", "ticket")
        identifier(ticket["ticket_id"], "ticket")
        require(ticket["ticket_id"] not in seen, "duplicate ticket")
        seen.add(ticket["ticket_id"])
        resident(ticket, region)
        require(ticket["subscriber_id"] == account["subscriber_id"], "ticket subscriber mismatch")
        require(ticket["category"] in ("network", "billing", "general"), "invalid ticket category")
        require(ticket["status"] in ("open", "resolved"), "invalid ticket status")
        adjustment = ticket["billing_adjustment"]
        if adjustment is not None:
            fields(adjustment, "amount reason residency", "billing adjustment")
            resident(adjustment, region)
            number(adjustment["amount"], "adjustment amount")
            text(adjustment["reason"], "billing adjustment reason")
            require(ticket["category"] == "billing", "adjustment needs billing ticket")
    for message in value["chat"]:
        fields(message, "role text residency", "chat message")
        resident(message, region)
        require(message["role"] in ("subscriber", "agent"), "invalid chat role")
        text(message["text"], "chat text")
    seen = set()
    for product in value["catalog"]:
        fields(product, "product_id residency monthly_cost data_gb", "product")
        resident(product, region)
        identifier(product["product_id"], "plan")
        require(product["product_id"] not in seen, "duplicate catalog product")
        seen.add(product["product_id"])
        number(product["monthly_cost"], "monthly cost")
        number(product["data_gb"], "plan data GB")
    require(account["current_plan"] in seen, "current plan missing from catalog")
    usage_records(value["usage_csv"], account["subscriber_id"], region)
    return value


def validate_state(state, phase):
    stages = ("prepared", "support", "onboard", "recommend")
    require(phase in stages, "invalid validation phase")
    index = stages.index(phase)
    expected = "schema_version synthetic residency stage subscriber usage_summary facts catalog"
    expected += " support" if index >= 1 else ""
    expected += " onboarding" if index >= 2 else ""
    expected += " recommendations" if index >= 3 else ""
    fields(state, expected, "pipeline state")
    require(state["stage"] == phase, "unexpected stage")
    region = state["residency"]
    account = state["subscriber"]
    fields(account, "subscriber_id residency discovery_consent current_plan goal completed_steps", "safe subscriber")
    identifier(account["subscriber_id"], "sub")
    require(type(account["discovery_consent"]) is bool, "invalid safe consent")
    require(account["goal"] in ("activate", "reduce_cost", "more_data"), "invalid safe goal")
    require(isinstance(account["completed_steps"], list) and
            all(s in ("activate_sim", "check_connection", "review_usage") for s in account["completed_steps"]),
            "invalid safe completed steps")
    summary = state["usage_summary"]
    fields(summary, "residency record_count call_minutes data_gb", "usage summary")
    require(type(summary["record_count"]) is int and summary["record_count"] >= 0, "invalid record count")
    number(summary["call_minutes"], "summary minutes")
    number(summary["data_gb"], "summary data")
    facts = state["facts"]
    fields(facts, "residency open_network_tickets adjustment_requests intent", "safe facts")
    require(isinstance(facts["open_network_tickets"], list), "invalid open tickets")
    for ticket in facts["open_network_tickets"]:
        identifier(ticket, "ticket")
    require(type(facts["adjustment_requests"]) is int and facts["adjustment_requests"] >= 0,
            "invalid adjustment count")
    require(facts["intent"] in ("network", "billing", "general"), "invalid intent")
    require(isinstance(state["catalog"], list), "invalid safe catalog")
    for product in state["catalog"]:
        fields(product, "product_id residency monthly_cost data_gb", "safe product")
        identifier(product["product_id"], "plan")
        number(product["monthly_cost"], "safe cost")
        number(product["data_gb"], "safe allowance")
        resident(product, region)
    require(account["current_plan"] in [p["product_id"] for p in state["catalog"]], "missing current plan")
    for record in (account, summary, facts):
        resident(record, region)
    if index >= 1:
        require(state["support"] == support_result(state), "invalid or ungrounded support output")
    if index >= 2:
        require(state["onboarding"] == onboarding_result(state), "invalid onboarding handoff")
    if index >= 3:
        require(state["recommendations"] == recommendation_result(state), "invalid recommendation handoff")


def prepare(source):
    validate(source)
    account = source["subscriber"]
    subscriber_messages = " ".join(m["text"].lower() for m in source["chat"] if m["role"] == "subscriber")
    intent = "network" if any(w in subscriber_messages for w in ("signal", "network", "connection")) else (
        "billing" if any(w in subscriber_messages for w in ("bill", "charge", "refund")) else "general")
    state = {
        "schema_version": "1.0", "synthetic": True, "residency": source["residency"], "stage": "prepared",
        "subscriber": {k: copy.deepcopy(account[k]) for k in
                       ("subscriber_id", "residency", "discovery_consent", "current_plan", "goal", "completed_steps")},
        "usage_summary": usage_records(source["usage_csv"], account["subscriber_id"], source["residency"]),
        "facts": {"residency": source["residency"],
                  "open_network_tickets": sorted(t["ticket_id"] for t in source["tickets"]
                                                 if t["category"] == "network" and t["status"] == "open"),
                  "adjustment_requests": sum(t["billing_adjustment"] is not None for t in source["tickets"]),
                  "intent": intent},
        "catalog": copy.deepcopy(source["catalog"]),
    }
    return validate(state, "prepared")


def support_result(state):
    facts = state["facts"]
    blocked = bool(facts["open_network_tickets"])
    if blocked:
        answer = "An open network ticket is recorded. Restart your device and check its connection. A human follow-up is needed; no repair time is confirmed."
        action = "follow_up_network"
    elif facts["adjustment_requests"] or facts["intent"] == "billing":
        answer = "Review your itemized bill against usage. Any requested adjustment requires human review; no credit has been applied."
        action = "review_billing"
    elif facts["intent"] == "network":
        answer = "Restart your device and check its connection. No open network fault is recorded here; persistent issues need a new support ticket."
        action = "check_connection"
    else:
        answer = "I can help with activation, connection checks, and usage review. Follow your next onboarding step; account-specific issues can be sent for human review."
        action = "continue_onboarding"
    return {"residency": state["residency"], "answer": answer, "action": action,
            "blocked": blocked, "source_ticket_ids": facts["open_network_tickets"],
            "billing_adjustments_applied": False}


def onboarding_result(state):
    support = state["support"]
    account = state["subscriber"]
    if support["blocked"]:
        step, instruction = "await_network", "Follow up the open network ticket before activating or changing service."
    else:
        order = ["activate_sim", "check_connection", "review_usage"]
        if account["goal"] == "reduce_cost" or support["action"] == "review_billing":
            order = ["review_usage", "activate_sim", "check_connection"]
        step = next((s for s in order if s not in account["completed_steps"]), "complete")
        instruction = {
            "activate_sim": "Activate your SIM through the subscriber account portal.",
            "check_connection": "Place a test call and verify a data connection.",
            "review_usage": "Compare your recorded usage with your plan and itemized bill.",
            "complete": "Onboarding is complete; review suitable plans when ready.",
        }[step]
    return {"residency": state["residency"], "source_support_action": support["action"],
            "blocked": support["blocked"], "next_step": step, "instruction": instruction,
            "goal": account["goal"]}


def recommendation_result(state):
    onboarding, account = state["onboarding"], state["subscriber"]
    reason = "matched_usage_and_goal"
    items = []
    if onboarding["blocked"]:
        reason = "deferred_network_fault"
    elif not account["discovery_consent"]:
        reason = "discovery_consent_not_granted"
    elif onboarding["next_step"] == "activate_sim":
        reason = "finish_activation_first"
    else:
        current = next(p for p in state["catalog"] if p["product_id"] == account["current_plan"])
        candidates = [p for p in state["catalog"] if p["product_id"] != current["product_id"]
                      and p["data_gb"] >= state["usage_summary"]["data_gb"]]
        if onboarding["goal"] == "reduce_cost":
            candidates = [p for p in candidates if p["monthly_cost"] < current["monthly_cost"]]
        if onboarding["goal"] == "more_data":
            candidates = [p for p in candidates if p["data_gb"] > current["data_gb"]]
        candidates.sort(key=lambda p: (p["monthly_cost"], p["data_gb"], p["product_id"]))
        items = [{"residency": state["residency"], "product_id": p["product_id"],
                  "monthly_cost": p["monthly_cost"], "data_gb": p["data_gb"]} for p in candidates[:3]]
        if not items:
            reason = "no_suitable_alternative"
    return {"residency": state["residency"], "source_onboarding_step": onboarding["next_step"],
            "reason": reason, "items": items}


def advance(state, previous, phase, field, builder):
    validate(state, previous)
    result = copy.deepcopy(state)
    result[field] = builder(result)
    result["stage"] = phase
    return validate(result, phase)


def support(state):
    return advance(state, "prepared", "support", "support", support_result)


def onboard(state):
    return advance(state, "support", "onboard", "onboarding", onboarding_result)


def recommend(state):
    return advance(state, "onboard", "recommend", "recommendations", recommendation_result)


def run_pipeline(source):
    return {"status": "ok", "result": recommend(onboard(support(prepare(source))))}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "provide exactly one input JSON path")
        with open(args[0], encoding="utf-8") as handle:
            source = json.load(handle, object_pairs_hook=unique_object)
        result = run_pipeline(source)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Never echo raw chat, identifiers, filenames, or exception payloads.
        print(json.dumps({"status": "error", "message": "Input file or schema validation failed."}))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
