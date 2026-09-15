from token_yield.wave3_oracles import (
    evaluate_oracle,
    normalize_text,
    normalize_value,
    validate_schema,
    validate_semantic_facts,
)


ORACLE = {
    "schema": {
        "type": "object",
        "required": ["agency", "publication_date", "amount", "status", "cited_ids"],
        "properties": {
            "agency": {"type": "string"},
            "publication_date": {"type": "string"},
            "amount": {"type": "string"},
            "status": {"type": "string", "enum": ["approved", "denied"]},
            "cited_ids": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    "required_facts": [
        {"field": "agency", "pattern": r"board of governors",
         "aliases": ["Federal Reserve & Board"]},
        {"field": "publication_date", "pattern": r"2024-03-29"},
        {"field": "amount", "pattern": r"1234\.5"},
    ],
    "cited_ids_field": "cited_ids",
    "input_ids": ["doc-1", "doc-2"],
}


def test_normalization_handles_unicode_aliases_dates_currency_and_space():
    text = normalize_text("  FEDERAL\u00a0Reserve & Board — March 29, 2024; $1,234.50! ")
    assert text == "federal reserve and board 2024-03-29 1234.5"
    assert normalize_text("29 March 2024") == "2024-03-29"
    assert normalize_text("03/29/2024") == "2024-03-29"
    assert normalize_value({"Name": ["A & B"]}) == {"Name": ["a and b"]}


def test_schema_required_types_enums_and_extra_fields():
    value = {
        "agency": "x", "publication_date": "x", "amount": "x",
        "status": "pending", "cited_ids": "doc-1", "extra": True,
    }
    errors = validate_schema(value, ORACLE["schema"])
    assert any("enum" in error for error in errors)
    assert any("expected array" in error for error in errors)
    assert any("additional field" in error for error in errors)


def test_semantic_facts_use_frozen_patterns_and_aliases():
    accepted, missing = validate_semantic_facts(
        {"agency": "Federal Reserve & Board",
         "publication_date": "March 29, 2024", "amount": "$1,234.50"},
        ORACLE["required_facts"],
    )
    assert accepted and missing == []


def test_complete_response_passes_all_independent_gates():
    output = """```json
    {"agency":"Board of Governors","publication_date":"March 29, 2024",
     "amount":"$1,234.50","status":"approved","cited_ids":["doc-1"]}
    ```"""
    result = evaluate_oracle(output, ORACLE, telemetry_complete=True)
    assert result == {
        "telemetry_complete": True, "provider_incomplete": False,
        "structural_acceptance": True, "semantic_acceptance": True,
        "overall_acceptance": True, "errors": [],
    }


def test_citations_must_be_subset_of_frozen_input_ids():
    value = {
        "agency": "Board of Governors", "publication_date": "2024-03-29",
        "amount": "1234.5", "status": "approved",
        "cited_ids": ["doc-1", "invented"],
    }
    result = evaluate_oracle(value, ORACLE, telemetry_complete=True)
    assert result["structural_acceptance"] is True
    assert result["semantic_acceptance"] is False
    assert result["overall_acceptance"] is False
    assert "invented" in result["errors"][0]


def test_truncation_and_missing_telemetry_are_not_wrong_content():
    truncated = evaluate_oracle("{", ORACLE, telemetry_complete=True,
                                provider_incomplete=True)
    assert truncated["provider_incomplete"] is True
    assert truncated["structural_acceptance"] is None
    assert truncated["semantic_acceptance"] is None
    assert truncated["overall_acceptance"] is False

    no_usage = evaluate_oracle("{}", ORACLE, telemetry_complete=False)
    assert no_usage["telemetry_complete"] is False
    assert no_usage["provider_incomplete"] is False
    assert no_usage["structural_acceptance"] is None
    assert no_usage["semantic_acceptance"] is None


def test_structural_failure_does_not_claim_semantic_failure():
    result = evaluate_oracle('{"agency": 3}', ORACLE, telemetry_complete=True)
    assert result["structural_acceptance"] is False
    assert result["semantic_acceptance"] is None
    assert result["overall_acceptance"] is False


def test_frozen_exact_values_are_normalized_before_comparison():
    oracle = {
        "schema": {
            "type": "object",
            "required": ["agency", "count"],
            "properties": {
                "agency": {"type": "string"},
                "count": {"type": "integer"},
            },
        },
        "equals": {"agency": "Federal Reserve & Board", "count": 3},
    }
    accepted = evaluate_oracle(
        {"agency": " federal reserve and board ", "count": 3},
        oracle,
        telemetry_complete=True,
    )
    assert accepted["semantic_acceptance"] is True
    rejected = evaluate_oracle(
        {"agency": "Federal Reserve", "count": 4},
        oracle,
        telemetry_complete=True,
    )
    assert rejected["semantic_acceptance"] is False
    assert len(rejected["errors"]) == 2
