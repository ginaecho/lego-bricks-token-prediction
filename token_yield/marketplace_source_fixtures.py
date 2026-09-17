"""Original fictional source data, separate from the immutable three-case registry."""

from copy import deepcopy

ARCHIVE_V2 = {
    "id": "archive-exceptions-v2",
    "version": "2.0.0",
    "title": "Archive Atelier v2 / fictional policy laboratory",
    "repository": "paperless-ngx/paperless-ngx",
    "url": "https://github.com/paperless-ngx/paperless-ngx",
    "observed": "2026-09-17",
    "metadata_url": "https://api.github.com/repos/paperless-ngx/paperless-ngx",
    "metadata_summary": "Document-management theme from the original scenario's metadata research only.",
    "inspiration": "Original invented demo policies, not Paperless policies or legal advice. "
    "Only the document-management theme is repository-inspired; no source or policy text copied.",
    "source_origin": "Original invented demonstration policies, not Paperless policies.",
    "description": "Archive Atelier v2 is an explicitly fictional policy laboratory. "
    "From its bundled v2 policy, record register and evidence notes, build a retention "
    "exception ledger: extract stated obligations and periods, classify exceptions using "
    "the supplied taxonomy, verify supporting passages and summarize missing evidence. "
    "Record the record ID, class, clock-start event/date, stated period, exception codes, "
    "owner, supporting document IDs and exact quotes, and unresolved questions. Mark "
    "unstated dates or rules unknown and preserve conflicting statements for review. "
    "Do not invent missing approvals, resolve ambiguity by assumption, or perform deletion. "
    "All rules and records are invented demonstration data, not Paperless policies, "
    "real legal/security authority, legal advice or a compliance assessment.",
    "new_function": "Retention exception ledger",
    "exercise": "Versioned source-alignment test; novelty, dissent and forecasts are not predetermined.",
}

_POLICY = """Fictional Archive Atelier policy AA-POL-2.0: retain invoice records for 24 calendar months after invoice closure.
Retain correspondence for 90 elapsed days after case closure; retain master media for 12 calendar months after project completion.
A start date is required to calculate a scheduled end. Missing starts and unscheduled record classes remain unknown.
Calendar-month periods use the same day in the target month, or its last day if unavailable. Dates are ISO calendar dates.
A documented internal review hold suspends disposal until a release note identifies the same record. It does not replace the base period.
An extension requires a record-specific approval naming the approving owner and explicit end date. An unsigned request alone is not approval.
The register owner must assemble supporting passages and refer unresolved exceptions for human review; this demo performs no disposal.
Taxonomy AA-TAX-2.0 permits multiple codes per record:
HOLD_ACTIVE: a documented hold has no documented release.
START_UNKNOWN: the event or date starting the stated period is absent.
CLASS_UNSCHEDULED: no supplied policy period covers the stated record class.
CONFLICTING_EVIDENCE: supplied statements disagree on class, clock date, owner or disposition rule.
APPROVAL_MISSING: an extension is requested but a named-owner approval or explicit end date is absent.
EARLY_DISPOSAL_REPORTED: a supplied note reports disposal before a calculable base end, with no supplied exception explaining it.
OWNER_UNKNOWN: no accountable owner is documented.
APPROVED_EXTENSION: a record-specific named-owner approval states an explicit extended end date.
No exception can be claimed absent supporting passages; no applicable exception may be reported as such only within supplied evidence.
All policy obligations above are invented internal demo rules. They are not Paperless rules, external legal/security authority, legal advice or a compliance determination."""

_CASES = (
    (
        "INV-10 is an invoice closed 2024-02-15, owner Ari. A review hold was recorded 2026-01-10.\n"
        "INV-11 is an invoice, owner Ari; the register does not state its closure date.\n"
        "MEDIA-12 is master media with project completion 2025-10-30; owner is not documented.",
        "Note N-10 confirms the hold for INV-10. No release note is supplied.\n"
        "Note N-11 requests a 6-month extension for INV-11, without a signature or end date.\n"
        "The register contains no subsequent evidence about MEDIA-12 ownership.",
    ),
    (
        "COR-20 is correspondence for a case closed 2026-05-11, owner Bo.\n"
        "INV-21 is an invoice closed 2024-01-31, owner Bo.\n"
        "PHOTO-22 is a customer photograph collection closed 2026-04-01, owner Bo.",
        "Note N-20 reports COR-20 disposed 2026-05-15. No exception approval is supplied.\n"
        "Approval AP-21 by Bo extends INV-21 through 2027-12-31 and identifies INV-21 explicitly.\n"
        "No schedule for customer photograph collections is supplied.",
    ),
    (
        "INV-30 is an invoice closed 2024-05-15, owner Chen.\n"
        "MEDIA-31 is master media with project completion 2025-07-01, owner Chen.",
        "Note N-30 states INV-30 closed 2024-06-15 instead. No correction notice resolves the two dates.\n"
        "Request R-31 asks to retain MEDIA-31 longer. It names no approving owner or end date.\n"
        "An unsigned memo says invoices should be kept seven years but cites no policy revision or approval.",
    ),
    (
        "COR-40 is correspondence, owner Dara; its case-closure event is not documented.\n"
        "INV-41 is an invoice closed 2024-02-29, owner Dara.",
        "Hold H-41 for INV-41 was recorded 2026-01-01. Release REL-41 explicitly releases INV-41 on 2026-02-01.\n"
        "No extension for INV-41 is supplied. Do not treat the released hold as still active.\n"
        "A system-import timestamp of 2026-03-01 exists for COR-40, but the note does not call it case closure.",
    ),
    (
        "MEDIA-50 is master media with project completion 2025-08-31, owner Eli.\n"
        "INV-51 is an invoice; closure date and owner are not documented.",
        "Hold H-50 identifies MEDIA-50; a release refers to MEDIA-05, not MEDIA-50. No correction is supplied.\n"
        "Request R-51 asks to retain INV-51 until 2028-01-01, but approving owner is missing.\n"
        "These identifiers must not be assumed equivalent.",
    ),
    (
        "COR-60 is correspondence closed 2026-01-01, owner Fran.\n"
        "MEDIA-61 is classified as master media, project completed 2025-12-31, owner Fran.",
        "Note N-60 reports COR-60 disposed 2026-01-10. No explaining exception is supplied.\n"
        "Note N-61 instead classifies MEDIA-61 as customer photographs and names owner Eli.\n"
        "No reconciliation of MEDIA-61 class or owner is supplied; preserve both statements.",
    ),
)


def fixture_scenario(fixture_id: str) -> dict:
    """Return independent metadata; only the explicitly versioned fixture is supported."""
    if fixture_id != ARCHIVE_V2["id"]:
        raise ValueError(f"Unknown source fixture: {fixture_id}")
    return deepcopy(ARCHIVE_V2)


def fixture_documents(fixture_id: str, group: str, index: int) -> list[dict]:
    """Build linked, source-only documents with six distinct evidence cases."""
    fixture_scenario(fixture_id)
    if type(index) is not int or not 0 <= index < len(_CASES):
        raise ValueError("Source fixture index must be an integer from 0 through 5")
    register, notes = _CASES[index]
    prefix = f"{fixture_id}-{group}"
    documents = (
        ("Fictional policy and exception taxonomy AA-POL-2.0", _POLICY),
        ("Fictional record register AA-REG-2.0", register),
        ("Fictional evidence notes AA-NOTES-2.0", notes),
    )
    return [
        {"id": f"{prefix}-doc{number}", "title": title,
         "text": f"{text}\nSource group: {group}. Fictional demonstration records only; not legal advice.",
         "role": "reference" if number == 0 else "project",
         "links": [f"{prefix}-doc{(number + 1) % 3}"]}
        for number, (title, text) in enumerate(documents)
    ]
