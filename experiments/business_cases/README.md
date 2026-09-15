# Business-case evidence

`cases.jsonl` is the first reproducible base-brick catalog. It contains nine
atomic tasks (one per primitive) and three compositions across finance,
reporting, support, legal, and procurement.

Each case has:

- an official HTTPS source showing that the data shape or workflow is real;
- a deterministic, created fixture so the expected result cannot drift;
- explicit primitive counts, size driver, and context bytes;
- acceptance rules that are evaluated independently of token cost; and
- a `held_out` flag for cases excluded from fitting.

The fixtures are **created probes**, not claims that a named organization ran
these exact prompts. Their official sources establish external validity; their
embedded facts establish reproducibility.

`proof_results.jsonl` records acceptance proofs. A proof without provider usage
is intentionally acceptance-only and must not enter token economics. Live runs
must use `token_yield.runner.run_case`, which requires measured token usage and
records model/version, prompt and source-manifest hashes, time, tool uses, and
the acceptance verdict.

Run the evidence report:

```bash
python -m examples.business_case_demo
```

Add `--blended-price` or `--split-price` only with the actual rate paid. The
demo will not invent prices or compute cost from acceptance-only proofs.
