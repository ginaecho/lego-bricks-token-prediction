# Northwind supplier invoice intake

Northwind Manufacturing wants to stop keying supplier invoices by hand. Their
accounts payable team of six currently processes about 24,000 invoices a month
across roughly 900 suppliers, and the backlog at month end is costing them
early-payment discounts.

For each invoice the pipeline should read the document, extract nine header and
line fields (supplier, invoice number, date, PO reference, net, tax, gross,
currency, payment terms), classify it to the right approval queue, check the
totals against the matching purchase order, and correct the simple exceptions —
transposed digits, missing PO prefix — without a human.

Anything that fails the purchase-order check goes to a person. Invoices average
about 2 kB of extracted text. The business case is straight cost reduction: they
want the AP headcount redeployed, not grown.

Team: we have committed a solution architect, two software engineers and a
project manager. Northwind's own AP lead will handle change management, so we
are not staffing that.
