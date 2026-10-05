"""Synthetic financial semantic search reference; no compliance certification.

CLI: python -B implementation.py example_input.json
Library: search(request, embed=None), where embed(list[str]) returns same-length
finite, nonzero numeric vectors of a consistent dimension. No provider is used.
"""

import csv
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def text(value, label, limit=10000):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit,
            label + " must be a nonempty bounded string")
    return value


def number(value, label):
    require(type(value) in (int, float), label + " must be numeric")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid, label + " must be finite")
    return value


CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def mask_string(value):
    return CARD.sub(lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], value)


def safe(value):
    if isinstance(value, str):
        return mask_string(value)
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {mask_string(key): safe(item) for key, item in value.items()}
    return value


def card(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9 -]+", value) is not None,
            "card_number must contain only digits and separators")
    digits = re.sub(r"\D", "", value)
    require(13 <= len(digits) <= 19, "card_number must have 13 to 19 digits")
    return "****" + digits[-4:]


def money(data):
    require(isinstance(data, dict), "financial payload must be an object")
    amount = number(data.get("amount"), "amount")
    require(amount > 0, "amount must be positive")
    currency = data.get("currency")
    require(isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency),
            "currency must be three uppercase letters")
    result = {"amount": amount, "currency": currency,
              "customer_id": text(data.get("customer_id"), "customer_id", 100)}
    if "card_number" in data:
        result["card_number"] = card(data["card_number"])
    return result


def transaction(data):
    require(isinstance(data, dict), "transaction data must be an object")
    fmt, payload = data.get("format"), data.get("payload")
    require(fmt in ("json", "csv", "iso20022"), "unsupported transaction format")
    if fmt == "csv":
        text(payload, "CSV payload")
        try:
            reader = csv.DictReader(io.StringIO(payload), strict=True)
            fields = reader.fieldnames
            require(fields and len(fields) == len(set(fields)), "CSV headers must be unique")
            rows = list(reader)
            require(len(rows) == 1 and None not in rows[0]
                    and all(v is not None for v in rows[0].values()),
                    "each transaction CSV payload must contain exactly one complete row")
            payload = rows[0]
            payload["amount"] = float(payload.get("amount", ""))
        except (csv.Error, ValueError) as exc:
            raise ValidationError("invalid transaction CSV") from exc
    elif fmt == "iso20022":
        try:
            payment = payload["Document"]["FIToFICstmrCdtTrf"]["CdtTrfTxInf"]
            amount = payment["IntrBkSttlmAmt"]
            payload = {"amount": amount["value"], "currency": amount["Ccy"],
                       "customer_id": payment["Dbtr"]["Id"]}
            payment_id = text(payment["PmtId"]["EndToEndId"], "payment ID", 100)
            iban = text(payment["DbtrAcct"]["Id"]["IBAN"], "IBAN", 100)
        except (KeyError, TypeError) as exc:
            raise ValidationError("invalid ISO 20022-style payment structure") from exc
    result = money(payload)
    result["source_format"] = fmt
    if fmt == "iso20022":
        result.update(payment_id=payment_id, iban=iban)
    return result


def validate(request):
    """All input formats enter the same normalized record validation layer."""
    require(isinstance(request, dict), "request must be an object")
    require(type(request.get("schema_version")) is int
            and request["schema_version"] == 1, "schema_version must be 1")
    require(request.get("synthetic") is True, "synthetic must be true")
    query = text(request.get("query"), "query", 1000)
    rows = request.get("records")
    require(isinstance(rows, list) and len(rows) <= 1000, "records must be a list of at most 1000")
    options = request.get("options", {})
    require(isinstance(options, dict), "options must be an object")
    require(not set(options) - {"top_k", "include_flagged"}, "unknown search option")
    top_k = options.get("top_k", 5)
    include = options.get("include_flagged", False)
    require(type(top_k) is int and 1 <= top_k <= 100, "top_k must be an integer in 1..100")
    require(type(include) is bool, "include_flagged must be boolean")
    records, seen = [], set()
    for row in rows:
        require(isinstance(row, dict), "each record must be an object")
        ident = text(row.get("id"), "record ID", 100)
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,99}", ident) is not None
                and not CARD.search(ident), "record ID must be a safe symbolic identifier")
        require(ident not in seen, "record IDs must be unique")
        seen.add(ident)
        entity = row.get("entity")
        require(entity in ("customer", "transaction", "loan_application"), "unknown entity")
        description = text(row.get("text"), "record text")
        kyc = row.get("kyc_status")
        require(kyc in ("verified", "pending", "rejected"), "invalid KYC status")
        flags = row.get("aml_flags")
        require(isinstance(flags, list) and len(flags) <= 20, "aml_flags must be a bounded list")
        flags = [text(flag, "AML flag", 100) for flag in flags]
        data = row.get("data")
        require(isinstance(data, dict), "record data must be an object")
        if entity == "transaction":
            normalized = transaction(data)
        elif entity == "loan_application":
            normalized = money(data)
            status = data.get("status")
            require(status in ("submitted", "review", "approved", "declined"), "invalid loan status")
            normalized["status"] = status
        else:
            normalized = {key: text(data.get(key), key, 100)
                          for key in ("full_name", "account_id", "iban")}
            if "card_number" in data:
                normalized["card_number"] = card(data["card_number"])
        records.append(safe({"id": ident, "entity": entity, "text": description,
                             "kyc_status": kyc, "aml_flags": flags, "data": normalized,
                             "screening_required": kyc != "verified" or bool(flags)}))
    return mask_string(query), records, top_k, include


GROUPS = (
    ("loan", "loans", "mortgage", "lending", "credit"),
    ("payment", "payments", "transfer", "transfers", "remittance"),
    ("kyc", "identity", "verification"),
    ("aml", "suspicious", "screening"),
    ("customer", "customers", "client", "clients"),
)
SYNONYMS = {word: group[0] for group in GROUPS for word in group}


def tokens(value):
    return [SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", value.lower())]


def unit(values):
    scale = max((abs(value) for value in values), default=0)
    require(scale > 0, "embedding vectors must not be zero")
    scaled = [value / scale for value in values]
    norm = math.sqrt(sum(value * value for value in scaled))
    return [value / norm for value in scaled]


def embeddings(embed, texts):
    require(callable(embed), "embed must be callable")
    try:
        result = embed(list(texts))
    except Exception as exc:
        raise ValidationError("injected embedding callable failed") from exc
    require(isinstance(result, (list, tuple)) and len(result) == len(texts),
            "embedding batch size mismatch")
    dimension, normalized = None, []
    for vector in result:
        require(isinstance(vector, (list, tuple)) and 1 <= len(vector) <= 4096,
                "embedding vectors must have 1..4096 dimensions")
        dimension = len(vector) if dimension is None else dimension
        require(len(vector) == dimension, "embedding dimension mismatch")
        normalized.append(unit([number(v, "embedding component") for v in vector]))
    return normalized


def search(request, embed=None):
    query, records, top_k, include = validate(request)
    eligible = [row for row in records if include or not row["screening_required"]]
    documents = [row["text"] + " " + row["entity"] + " "
                 + json.dumps(row["data"], sort_keys=True) for row in eligible]
    bags = [Counter(tokens(doc)) for doc in documents]
    index = defaultdict(list)
    for position, bag in enumerate(bags):
        for term in bag:
            index[term].append(position)
    query_bag = Counter(tokens(query))
    idf = {term: math.log((len(bags) + 1) / (len(index.get(term, [])) + 1)) + 1
           for term in set(index) | set(query_bag)}

    def weighted(bag):
        values = {term: count * idf[term] for term, count in bag.items()}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {term: value / norm for term, value in values.items()} if norm else {}

    qvec = weighted(query_bag)
    vectors = embeddings(embed, [query] + documents) if embed is not None else None
    candidates = set(range(len(eligible))) if vectors is not None else {
        position for term in query_bag for position in index.get(term, [])}
    hits = []
    for position in sorted(candidates):
        dvec = weighted(bags[position])
        contributions = {term: qvec[term] * dvec[term]
                         for term in sorted(set(qvec) & set(dvec))}
        lexical = sum(contributions.values())
        cosine = sum(a * b for a, b in zip(vectors[0], vectors[position + 1])) if vectors else 0.0
        similarity = max(0.0, min(1.0, cosine))
        weight = 0.35 if vectors else 0.0
        score = (1 - weight) * lexical + weight * similarity
        if score <= 0:
            continue
        hits.append({"record": eligible[position], "score": score, "explanation": {
            "method": "synonym-normalized TF-IDF cosine with optional embedding cosine",
            "canonical_term_contributions": contributions,
            "lexical_score": lexical, "embedding_cosine": cosine,
            "embedding_score": similarity, "lexical_weight": 1 - weight,
            "embedding_weight": weight,
            "formula": "lexical_weight * lexical_score + embedding_weight * embedding_score",
            "screening": "review_required" if eligible[position]["screening_required"] else "clear",
            "meaning": "Search relevance only; not creditworthiness, AML risk, or a lending decision."
        }})
    hits.sort(key=lambda hit: (-hit["score"], hit["record"]["id"]))
    return {"status": "ok", "schema_version": 1, "synthetic": True, "query": query,
            "results": hits[:top_k], "summary": {"indexed": len(eligible),
            "excluded_by_screening": len(records) - len(eligible), "matched": len(hits)},
            "notice": "Demonstrative KYC/AML flags and card masking only; no compliance certification."}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("nonfinite JSON number")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "usage: python -B implementation.py INPUT.json")
        with open(argv[0], encoding="utf-8") as handle:
            request = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = search(request)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else "unable to read valid input JSON"
        print(json.dumps({"status": "error", "error": message}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
