# Clinical coding audit sampling

Lamna's coding audit team pulls a monthly sample of discharge summaries and
checks the assigned diagnosis codes against the documentation. They sample 2% of
episodes because that is all they have capacity for, and the payer disputes
suggest the error rate in the other 98% is not the same.

The pipeline should read each discharge summary, extract the documented
diagnoses and procedures, classify each against the coding standard, reconcile
what was documented against what was actually coded and billed, and write up
each disagreement in the form the audit team already uses.

They want to move from a 2% sample to 100% coverage. That is about 18,000
episodes a month. Summaries are around 8 kB. Funded as compliance and revenue
integrity — under-coding is costing them as much as over-coding is risking them.
