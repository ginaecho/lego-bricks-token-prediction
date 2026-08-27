# Supplier non-conformance handling

Tailspin's quality team raises about 700 supplier non-conformance reports a
month. Each one currently takes an engineer half a day to work up before it can
be sent to the supplier.

The pipeline reads the inspection report and the associated drawing notes,
extracts the failed characteristics and their tolerances, classifies the failure
against the standard cause taxonomy, retrieves the previous non-conformances for
that supplier and part, reconciles the current failure against that history to
say whether it is a repeat, and drafts the report that goes to the supplier.

Cost reduction, and a quality argument: repeat failures are currently only
spotted when someone remembers them. Inspection reports plus drawing notes come
to roughly 12 kB.
