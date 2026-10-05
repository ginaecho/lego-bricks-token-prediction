# Outage report reconciliation

Relecloud Energy has to report unplanned outages to the regulator monthly, and
the numbers in the operational system have never quite agreed with the numbers
in the asset management system. Reconciling them is two people for a week every
month.

Per reporting cycle: read both extracts and the field engineers' notes, extract
the outage events with their start and end times and affected customer counts,
reconcile the two systems event by event, remediate the classes of mismatch that
have a known rule (clock drift, duplicate events raised at shift handover), and
report what is left for a human.

Monthly. Combined extracts and notes come to roughly 200 kB. Compliance driven —
the regulator has already written to them about the discrepancies.

Team: propose one. The reconciliation rules live in the field engineers'
heads and somebody has to get them out.
