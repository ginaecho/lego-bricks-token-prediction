"""Deterministic, standard-library-only onboarding-to-document pipeline."""

import csv
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def fields(value, names, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == set(names), f"{label} fields must be: {', '.join(names)}")


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    return value.strip()


def money(value, label):
    require(isinstance(value, str), f"{label} must be a decimal string")
    require(bool(re.fullmatch(r"\d{1,9}(?:\.\d{1,2})?", value)), f"{label} must be nonnegative money with at most two decimals")
    try:
        return Decimal(value).quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise ValidationError(f"Invalid {label}") from exc


def amount(value):
    return format(value, ".2f")


def strings(value, label):
    require(isinstance(value, list), f"{label} must be a list")
    normalized = [text(item, label).casefold() for item in value]
    require(len(normalized) == len(set(normalized)), f"{label} must be unique")
    return normalized


def validate(kind, value, source=None):
    """One validation boundary for input and every stage handoff."""
    if kind == "input":
        fields(value, ["schema_version", "fixture", "customer", "catalog", "document", "limit"], kind)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1, "Unsupported schema_version")
        require(value["fixture"] == "synthetic", "fixture must be labeled synthetic")
        c = value["customer"]
        fields(c, ["id", "name", "email", "interests", "budget", "completed_steps"], "customer")
        for key in ("id", "name"):
            text(c[key], f"customer.{key}")
        require(isinstance(c["email"], str), "email must be text")
        require(c["email"] == "" or bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", c["email"])), "Invalid email")
        strings(c["interests"], "interests")
        money(c["budget"], "budget")
        completed = strings(c["completed_steps"], "completed_steps")
        require(all(step in ("profile", "preferences", "verification") for step in completed), "Unknown completed step")
        require("profile" not in completed or c["email"] != "", "Completed profile needs email")
        require("preferences" not in completed or bool(c["interests"]), "Completed preferences need interests")
        require("verification" not in completed or "profile" in completed, "Verification needs a completed profile")
        require(isinstance(value["catalog"], list), "catalog must be a list")
        ids = set()
        for product in value["catalog"]:
            fields(product, ["id", "name", "tags", "price", "in_stock"], "product")
            pid = text(product["id"], "product.id")
            require(pid == product["id"] and pid not in ids, "Product IDs must be trimmed and unique")
            ids.add(pid)
            text(product["name"], "product.name")
            strings(product["tags"], "tags")
            money(product["price"], "price")
            require(type(product["in_stock"]) is bool, "in_stock must be boolean")
        require(type(value["limit"]) is int and 1 <= value["limit"] <= 20, "limit must be an integer from 1 to 20")
        fields(value["document"], ["format", "content"], "document")
        require(value["document"]["format"] == "csv", "Only CSV documents are supported")
        text(value["document"]["content"], "document.content")
    elif kind == "onboarding":
        fields(value, ["customer_id", "greeting", "next_step", "message", "interests", "budget"], kind)
        require(value["customer_id"] == source["customer"]["id"], "Customer handoff mismatch")
        require(value["next_step"] in ("profile", "preferences", "verification", "discover"), "Invalid next step")
        for key in ("greeting", "message"):
            text(value[key], key)
        require(value["interests"] == strings(source["customer"]["interests"], "interests"), "Interest handoff mismatch")
        require(money(value["budget"], "budget") == money(source["customer"]["budget"], "budget"), "Budget handoff mismatch")
    elif kind == "discovery":
        fields(value, ["customer_id", "onboarding_step", "budget", "items", "reason"], kind)
        require(value["customer_id"] == source["customer_id"], "Customer handoff mismatch")
        require(value["onboarding_step"] == source["next_step"], "Next-step handoff mismatch")
        require(value["budget"] == source["budget"], "Budget handoff mismatch")
        require(isinstance(value["items"], list), "items must be a list")
        seen = set()
        for item in value["items"]:
            fields(item, ["product_id", "name", "price", "score", "matched_interests", "reason"], "recommendation")
            require(item["product_id"] not in seen, "Duplicate recommendation")
            seen.add(text(item["product_id"], "product_id"))
            text(item["name"], "name")
            text(item["reason"], "reason")
            require(money(item["price"], "price") <= money(value["budget"], "budget"), "Recommendation exceeds budget")
            matches = strings(item["matched_interests"], "matched_interests")
            require(all(tag in source["interests"] for tag in matches), "Unexpected matched interest")
            require(type(item["score"]) is int and item["score"] == len(matches), "Invalid recommendation score")
        text(value["reason"], "reason")
    elif kind == "documents":
        fields(value, ["customer_id", "onboarding_step", "recommended_product_ids", "rows", "total", "checks", "valid"], kind)
        require(value["customer_id"] == source["customer_id"], "Customer handoff mismatch")
        require(value["onboarding_step"] == source["onboarding_step"], "Next-step handoff mismatch")
        require(value["recommended_product_ids"] == [p["product_id"] for p in source["items"]], "Recommendation handoff mismatch")
        require(isinstance(value["rows"], list), "rows must be a list")
        total = Decimal("0")
        for row in value["rows"]:
            fields(row, ["product_id", "quantity", "unit_price", "line_total", "recommended", "checks"], "row")
            text(row["product_id"], "row.product_id")
            require(type(row["quantity"]) is int and 1 <= row["quantity"] <= 10000, "Invalid row quantity")
            expected = money(row["unit_price"], "unit_price") * row["quantity"]
            require(row["line_total"] == amount(expected), "Invalid line total")
            require(type(row["recommended"]) is bool and row["recommended"] == (row["product_id"] in value["recommended_product_ids"]), "Invalid recommended flag")
            strings(row["checks"], "row checks")
            total += expected
        require(value["total"] == amount(total), "Invalid document total")
        strings(value["checks"], "document checks")
        require(type(value["valid"]) is bool and value["valid"] == (not value["checks"] and all(not r["checks"] for r in value["rows"])), "Invalid document validity")
    else:
        raise ValidationError("Unknown validation schema")
    return value


def onboard(data):
    validate("input", data)
    customer = data["customer"]
    completed = strings(customer["completed_steps"], "completed_steps")
    step = next((s for s in ("profile", "preferences", "verification") if s not in completed), "discover")
    messages = {
        "profile": "Complete your contact profile so we can support your first purchase.",
        "preferences": "Choose your interests to personalize product suggestions.",
        "verification": "Verify your contact details before completing onboarding.",
        "discover": "Explore the products selected for your interests and budget.",
    }
    result = {
        "customer_id": customer["id"],
        "greeting": f"Welcome, {customer['name'].strip()}!",
        "next_step": step,
        "message": messages[step],
        "interests": strings(customer["interests"], "interests"),
        "budget": amount(money(customer["budget"], "budget")),
    }
    return validate("onboarding", result, data)


def recommend(data, onboarding):
    validate("onboarding", onboarding, data)
    interests = set(onboarding["interests"])
    budget = money(onboarding["budget"], "budget")
    items = []
    for product in data["catalog"]:
        if not product["in_stock"] or money(product["price"], "price") > budget:
            continue
        matches = sorted(interests.intersection(strings(product["tags"], "tags")))
        items.append({
            "product_id": product["id"], "name": product["name"],
            "price": amount(money(product["price"], "price")),
            "score": len(matches), "matched_interests": matches,
            "reason": "Matches: " + ", ".join(matches) if matches else "Affordable in-stock alternative",
        })
    items.sort(key=lambda p: (-p["score"], Decimal(p["price"]), p["product_id"]))
    result = {
        "customer_id": onboarding["customer_id"], "onboarding_step": onboarding["next_step"],
        "budget": onboarding["budget"], "items": items[:data["limit"]],
        "reason": "Interest overlap, then price, then product ID" if items else "No in-stock products within budget",
    }
    return validate("discovery", result, onboarding)


def automate_documents(data, onboarding, discovery):
    validate("discovery", discovery, onboarding)
    catalog = {p["id"]: p for p in data["catalog"]}
    recommended = [p["product_id"] for p in discovery["items"]]
    rows = []
    total = Decimal("0")
    try:
        reader = csv.reader(io.StringIO(data["document"]["content"], newline=""), strict=True)
        header = next(reader, [])
        header = [h.strip().casefold() for h in header]
        require(len(header) == 3 and set(header) == {"sku", "quantity", "unit_price"}, "CSV needs unique sku, quantity, unit_price columns")
        for line_number, values in enumerate(reader, 2):
            if not values or all(not item.strip() for item in values):
                continue
            require(len(values) == 3, f"CSV row {line_number} has wrong column count")
            record = dict(zip(header, (v.strip() for v in values)))
            sku = text(record["sku"], f"row {line_number} sku")
            require(bool(re.fullmatch(r"\d{1,5}", record["quantity"])), f"row {line_number} quantity must be a positive integer")
            quantity = int(record["quantity"])
            require(1 <= quantity <= 10000, f"row {line_number} quantity must be 1..10000")
            price = money(record["unit_price"], f"row {line_number} unit_price")
            checks = []
            product = catalog.get(sku)
            if product is None:
                checks.append("unknown_product")
            else:
                if price != money(product["price"], "price"):
                    checks.append("price_mismatch")
                if not product["in_stock"]:
                    checks.append("out_of_stock")
            if sku not in recommended:
                checks.append("not_recommended")
            line_total = price * quantity
            total += line_total
            rows.append({
                "product_id": sku, "quantity": quantity, "unit_price": amount(price),
                "line_total": amount(line_total), "recommended": sku in recommended, "checks": checks,
            })
    except csv.Error as exc:
        raise ValidationError(f"Invalid CSV: {exc}") from exc
    checks = []
    if not rows:
        checks.append("empty_document")
    if total > money(discovery["budget"], "budget"):
        checks.append("over_budget")
    result = {
        "customer_id": discovery["customer_id"], "onboarding_step": discovery["onboarding_step"],
        "recommended_product_ids": recommended, "rows": rows, "total": amount(total),
        "checks": checks, "valid": not checks and all(not row["checks"] for row in rows),
    }
    return validate("documents", result, discovery)


def run_pipeline(data):
    validate("input", data)
    onboarding = onboard(data)
    discovery = recommend(data, onboarding)
    documents = automate_documents(data, onboarding, discovery)
    return {
        "schema_version": 1, "fixture": data["fixture"], "status": "ok",
        "customer_id": onboarding["customer_id"],
        "onboarding": onboarding, "discovery": discovery, "documents": documents,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 1, "Usage: python -B implementation.py example_input.json")
        with open(argv[0], encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        output = run_pipeline(data)
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
