"""Synthetic, deterministic discovery -> feedback -> offline web research.

Usage: python -B implementation.py example_input.json
Only supplied page snapshots are retrieved; this program never makes requests.
All schemas, bounds, grounding, and stage handoffs use validate().
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


VERSION = "1.0"
MAX_FILE_BYTES = 2_000_000
THEMES = {
    "quality": {"quality", "broken", "sturdy", "durable"},
    "shipping": {"shipping", "delivery", "late", "arrived"},
    "value": {"value", "price", "cheap", "expensive"},
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, location):
    require(isinstance(value, dict), location + " must be an object")
    require(set(value) == set(names.split()), location + " has missing or unknown fields")


def text(value, location, maximum=20000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            location + " must be a bounded nonblank string")


def array(value, location, maximum=1000):
    require(isinstance(value, list) and len(value) <= maximum,
            location + " must be a bounded array")


def strings(value, location):
    array(value, location)
    for element in value:
        text(element, location, 200)
    require(len(value) == len(set(value)), location + " must be unique")


def integer(value, lower, upper, location):
    require(type(value) is int and lower <= value <= upper,
            location + " must be an integer in range")


def words(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def canonical_url(value, allowlist):
    text(value, "page URL", 2000)
    require(not any(c.isspace() or ord(c) < 32 for c in value)
            and "\\" not in value, "ambiguous URL")
    try:
        parsed = urlsplit(value)
        require(parsed.scheme == "https" and parsed.hostname in allowlist,
                "URL must use HTTPS and an exact allowlisted host")
        require(parsed.username is None and parsed.password is None
                and parsed.port in (None, 443) and not parsed.fragment,
                "URL credentials, nonstandard ports, or fragments are forbidden")
        return urlunsplit(("https", parsed.hostname, parsed.path or "/", parsed.query, ""))
    except ValueError as exc:
        raise ValidationError("invalid URL: " + str(exc)) from exc


def normalized_feedback(value):
    return " ".join(value.casefold().split())


def eligible_items(source):
    profile = source["profile"]
    return [item for item in source["items"]
            if item["id"] not in profile["exclude_item_ids"]
            and not set(item["tags"]) & set(profile["exclude_tags"])]


def ranking(source):
    weights = source["profile"]["interests"]
    entries = []
    for item in eligible_items(source):
        reasons = [{"tag": tag, "weight": weights[tag]}
                   for tag in sorted(item["tags"]) if tag in weights]
        score = sum(reason["weight"] for reason in reasons)
        if score:
            entries.append({"item_id": item["id"], "score": score,
                            "reasons": reasons})
    return sorted(entries, key=lambda item: (-item["score"], item["item_id"]))[
        :source["profile"]["limit"]]


def feedback_groups(source, selected):
    groups = {}
    for entry in source["feedback"]:
        if entry["item_id"] in selected:
            key = (entry["item_id"], normalized_feedback(entry["text"]))
            if key not in groups:
                groups[key] = {"feedback_ids": [], "item_id": entry["item_id"],
                               "text": entry["text"]}
            groups[key]["feedback_ids"].append(entry["id"])
    return list(groups.values())


def classify(value):
    tokens = words(value)
    return [name for name, lexicon in THEMES.items() if tokens & lexicon] or ["other"]


def validate(kind, value, context=None):
    """Single shared boundary validator for input and every stage output."""
    if kind == "input":
        fields(value, "schema_version synthetic profile items feedback web", kind)
        require(value["schema_version"] == VERSION, "unsupported schema version")
        require(value["synthetic"] is True, "fixtures must be explicitly synthetic")
        profile = value["profile"]
        fields(profile, "interests exclude_tags exclude_item_ids limit", "profile")
        require(isinstance(profile["interests"], dict)
                and len(profile["interests"]) <= 100, "interests must be a bounded object")
        for tag, weight in profile["interests"].items():
            text(tag, "interest", 100)
            require(tag == tag.strip().casefold(), "interest tags must be normalized")
            integer(weight, 1, 100, "interest weight")
        strings(profile["exclude_tags"], "excluded tags")
        strings(profile["exclude_item_ids"], "excluded IDs")
        integer(profile["limit"], 1, 100, "recommendation limit")
        array(value["items"], "items")
        item_ids = set()
        for item in value["items"]:
            fields(item, "id name tags", "item")
            text(item["id"], "item ID", 200)
            text(item["name"], "item name", 200)
            strings(item["tags"], "item tags")
            require(all(tag == tag.strip().casefold() for tag in item["tags"]),
                    "item tags must be normalized")
            require(item["id"] not in item_ids, "duplicate item ID")
            item_ids.add(item["id"])
        require(set(profile["exclude_item_ids"]) <= item_ids, "unknown excluded item")
        require(all(tag == tag.strip().casefold() for tag in profile["exclude_tags"]),
                "excluded tags must be normalized")
        array(value["feedback"], "feedback")
        feedback_ids = set()
        for entry in value["feedback"]:
            fields(entry, "id item_id text", "feedback entry")
            text(entry["id"], "feedback ID", 200)
            text(entry["item_id"], "feedback item ID", 200)
            text(entry["text"], "feedback text")
            require(entry["item_id"] in item_ids, "feedback references unknown item")
            require(entry["id"] not in feedback_ids, "duplicate feedback ID")
            feedback_ids.add(entry["id"])
        web = value["web"]
        fields(web, "allowlist pages max_findings", "web")
        strings(web["allowlist"], "allowlist")
        for host in web["allowlist"]:
            require(bool(re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+",
                host)) and len(host) <= 253, "allowlist must contain lowercase DNS hosts")
        integer(web["max_findings"], 1, 100, "max findings")
        array(web["pages"], "pages")
        page_ids, urls = set(), set()
        for page in web["pages"]:
            fields(page, "id url title content", "page")
            text(page["id"], "page ID", 200)
            text(page["title"], "page title", 500)
            text(page["content"], "page content")
            url = canonical_url(page["url"], web["allowlist"])
            require(page["id"] not in page_ids and url not in urls,
                    "duplicate page ID or canonical URL")
            page_ids.add(page["id"])
            urls.add(url)
        return value
    require(context is not None, "validation context is required")
    source = context["input"]
    if kind == "interests":
        fields(value, "recommendations", kind)
        require(value["recommendations"] == ranking(source),
                "recommendations must match grounded, exclusion-safe ranking")
    elif kind == "feedback":
        fields(value, "selected_item_ids interest_terms analyzed_ids duplicate_count themes", kind)
        previous = validate("interests", context["interests"], {"input": source})
        selected = [entry["item_id"] for entry in previous["recommendations"]]
        interest_terms = sorted({reason["tag"] for entry in previous["recommendations"]
                                 for reason in entry["reasons"]})
        groups = feedback_groups(source, selected)
        expected_ids = [entry["id"] for entry in source["feedback"]
                        if entry["item_id"] in selected]
        require(value["selected_item_ids"] == selected
                and value["interest_terms"] == interest_terms
                and value["analyzed_ids"] == expected_ids,
                "feedback handoff does not match selected recommendations")
        integer(value["duplicate_count"], 0, 1000, "duplicate count")
        require(value["duplicate_count"] == len(expected_ids) - len(groups),
                "incorrect feedback deduplication count")
        expected_themes = {}
        for group in groups:
            for theme in classify(group["text"]):
                expected_themes.setdefault(theme, []).append(group)
        expected = [{"theme": theme, "count": len(excerpts), "excerpts": excerpts}
                    for theme, excerpts in sorted(expected_themes.items())]
        require(value["themes"] == expected, "ungrounded themes or feedback excerpts")
    elif kind == "web":
        fields(value, "queries findings", kind)
        previous = validate("feedback", context["feedback"], context)
        expected_queries = make_queries(previous)
        require(value["queries"] == expected_queries, "research queries lost feedback provenance")
        require(value["findings"] == retrieve(source, expected_queries),
                "findings must match allowlisted snapshots and traceable retrieval")
    else:
        raise ValidationError("unknown schema kind")
    return value


def recommend(source):
    validate("input", source)
    return validate("interests", {"recommendations": ranking(source)}, {"input": source})


def analyze_feedback(source, previous):
    validate("interests", previous, {"input": source})
    selected = [entry["item_id"] for entry in previous["recommendations"]]
    groups = feedback_groups(source, selected)
    themes = {}
    for group in groups:
        for theme in classify(group["text"]):
            themes.setdefault(theme, []).append(group)
    ids = [entry["id"] for entry in source["feedback"] if entry["item_id"] in selected]
    result = {
        "selected_item_ids": selected,
        "interest_terms": sorted({reason["tag"] for entry in previous["recommendations"]
                                  for reason in entry["reasons"]}),
        "analyzed_ids": ids,
        "duplicate_count": len(ids) - len(groups),
        "themes": [{"theme": theme, "count": len(excerpts), "excerpts": excerpts}
                   for theme, excerpts in sorted(themes.items())],
    }
    return validate("feedback", result, {"input": source, "interests": previous})


def make_queries(feedback):
    return [{"theme": entry["theme"],
             "terms": sorted(THEMES[entry["theme"]]),
             "interest_terms": feedback["interest_terms"],
             "feedback_ids": sorted({fid for excerpt in entry["excerpts"]
                                     for fid in excerpt["feedback_ids"]})}
            for entry in feedback["themes"] if entry["theme"] in THEMES]


def retrieve(source, queries):
    findings = []
    for query in queries:
        for page in source["web"]["pages"]:
            tokens = words(page["content"])
            matched = sorted(tokens & set(query["terms"]))
            if not matched:
                continue
            contextual = sorted({term for term in query["interest_terms"]
                                 if words(term) and words(term) <= tokens})
            # Findings are quotations, not claims of independent verification.
            findings.append({
                "theme": query["theme"], "feedback_ids": query["feedback_ids"],
                "page_id": page["id"],
                "url": canonical_url(page["url"], source["web"]["allowlist"]),
                "title": page["title"], "excerpt": page["content"],
                "start": 0, "end": len(page["content"]),
                "matched_terms": matched, "matched_interests": contextual,
                "score": len(matched) + len(contextual),
            })
    findings.sort(key=lambda entry: (-entry["score"], entry["theme"], entry["page_id"]))
    return findings[:source["web"]["max_findings"]]


def research(source, previous, interests):
    context = {"input": source, "interests": interests, "feedback": previous}
    validate("feedback", previous, context)
    queries = make_queries(previous)
    return validate("web", {"queries": queries, "findings": retrieve(source, queries)}, context)


def run_pipeline(source):
    validate("input", source)
    interests = recommend(source)
    feedback = analyze_feedback(source, interests)
    web = research(source, feedback, interests)
    return {"status": "ok", "schema_version": VERSION, "synthetic": True,
            "interests": interests, "feedback": feedback, "web": web}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON number: " + value)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, "usage: python -B implementation.py INPUT.json")
        with Path(args[0]).open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        require(len(raw) <= MAX_FILE_BYTES, "input file exceeds size limit")
        source = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
        result = run_pipeline(source)
        code = 0
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        result = {"status": "error", "error": str(exc)}
        code = 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
