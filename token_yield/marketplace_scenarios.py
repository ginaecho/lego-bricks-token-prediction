"""Original invented demo briefs inspired only by public repository metadata.

No repository source or customer documents are copied into provider prompts.
"""

SCENARIOS = [
    {
        "id": "archive-exceptions",
        "title": "Archive Manager",
        "repository": "paperless-ngx/paperless-ngx",
        "url": "https://github.com/paperless-ngx/paperless-ngx",
        "metadata_url": "https://api.github.com/repos/paperless-ngx/paperless-ngx",
        "observed": "2026-09-17",
        "metadata_summary": "Community document scanning, indexing and archiving.",
        "license_metadata": "GPL-3.0",
        "inspiration": "Document management inspired a retention-exception review.",
        "description": "A document-storage company keeps many documents and needs each one "
        "reviewed to work out how long it should be kept, with a list of the records where that "
        "is unclear or needs a manager to check.",
        "new_function": "Retention exception ledger",
        "exercise": "Explicit capability review and possible creation; agent verdict is not predetermined.",
    },
    {
        "id": "booking-exceptions",
        "title": "Booking Blocks",
        "repository": "calcom/cal.diy",
        "url": "https://github.com/calcom/cal.diy",
        "metadata_url": "https://api.github.com/repos/calcom/cal.com",
        "observed": "2026-09-17",
        "metadata_summary": "Scheduling infrastructure; cal.com API lookup redirected to cal.diy.",
        "license_metadata": "MIT",
        "inspiration": "Scheduling inspired a fictional booking-record retention register.",
        "description": "Invented Booking Blocks reviews retention exceptions for appointment records "
        "using only bundled fictional reference statements. A retention exception register should "
        "extract stated obligations, classify exceptions, verify supporting passages and write "
        "a concise ledger with missing evidence. Compare existing contracts before adding a brick; "
        "this register asks for the same bounded document transformation as a ledger.",
        "new_function": "Retention exception register",
        "exercise": "Similar reuse review after Archive Manager; lexical overlap is not proof.",
    },
    {
        "id": "support-handoffs",
        "title": "Care Crew",
        "repository": "chatwoot/chatwoot",
        "url": "https://github.com/chatwoot/chatwoot",
        "metadata_url": "https://api.github.com/repos/chatwoot/chatwoot",
        "observed": "2026-09-17",
        "metadata_summary": "Live-chat, email and omnichannel support desk.",
        "license_metadata": "NOASSERTION (GitHub metadata; no licensing interpretation)",
        "inspiration": "Support channels inspired fictional handoff obligation mapping.",
        "description": "Invented Care Crew coordinates support handoffs. From bundled fictional "
        "reference documents, map each stated handoff obligation to an owner, distinguish missing "
        "owners from conflicting statements, and plan two follow-up checks with quoted evidence. "
        "Detect absent functionality from this description, review whether existing contracts "
        "cover it, and establish a source-only contract only if genuinely new.",
        "new_function": "",
        "exercise": "Description-only detection enters the same automatic review path.",
    },
]


def scenario_for_request(request: dict) -> dict | None:
    """Identify an unchanged scenario without trusting a browser-supplied label."""
    return next((item for item in SCENARIOS
                 if request["description"] == item["description"]
                 and request.get("new_function", "") == item["new_function"]), None)
