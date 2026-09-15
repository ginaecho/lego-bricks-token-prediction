# Wave 3 aborted canary

This immutable canary run stopped after four logical cases and must not be used
for model selection.

- Two cases stopped before generation because the input-token endpoint rejected
  an empty input and then rejected the output-only `max_output_tokens` field.
- A third case stopped when Azure reported that this `gpt-5-mini` deployment is
  unsupported by the input-token endpoint.
- The fourth case generated successfully, but the Fetch prompt allowed the
  model to interpret `source_id` as the SEC CIK. The remaining extracted fields
  were correct, but the frozen structural oracle correctly rejected the output.
- The conservative safety ledger settled at USD 10.72512 cumulatively, including
  wave 2. This is an authorization guardrail, not an Azure billing claim.

The prompt ambiguity was corrected only in a new v2 preregistration. No record
in this directory was deleted, replaced, or relabeled.
