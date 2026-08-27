# Northwind warranty claim intake — phase 2

Follow-on to the supplier invoice pipeline we delivered for Northwind. Same
client, same team, same platform.

They now want the identical treatment for supplier warranty claims: read the
claim, pull out the same nine commercial fields plus the two warranty-specific
ones, classify to a handling queue, and reconcile the claimed amount against the
warranty terms held in the contract repository rather than against a purchase
order.

The extraction and routing are essentially what we already built. The
reconciliation is new — warranty terms are prose, not a structured PO line, so
that comparison has to be done properly rather than by field match.

About 9,000 claims a month. Documents are a bit longer than the invoices,
call it 3 kB each. Cost reduction again, and the client has already seen phase
one work, so the commercial conversation is about extending the existing
statement of work rather than a new one.
