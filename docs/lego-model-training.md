# How we trained the LEGO-like model

We trained a calculator to estimate how much text GPT reads and writes,
and the resulting API cost, when preparing a short project scope.
We did not retrain GPT or measure the cost of delivering a whole project.

## What are the bricks

We read six public customer requests and prepared 42 individual requirements.
Every request uses the same four tasks:

* Extract: copy the supplied requirements.
* Classify: copy the labels we already attached.
* Plan: suggest possible outputs and flag what needs human review.
* Report: summarize the scope and excluded work.

These are four tasks, not four industries. Each reads the same brief;
one task's answer is not passed to the next.
We measured them separately and combined into one GPT request.

## What the calculator learns

We reused 60 recorded GPT calls rather than inventing token counts.
Tokens are the small pieces of text GPT counts.
The calculator uses request length, task counts, allowed answer length,
and how tasks share requests.

Four projects taught it; two other projects tested it.
Repeated runs stayed with their project and did not become new projects.
We compared five prediction formulas, including using the same average for
every request. Separate formulas predict text read, text written and cost.

## What changed and how it performed

The new calculator predicts a complete scope preparation, not one call.
We later tested four more public requests without changing its formulas.

| Average size of a prediction mistake | Earlier two projects | Four new projects |
|---|---:|---:|
| Text sent to GPT | 186.8 tokens | 113.4 tokens |
| Text written by GPT | 48.8 tokens | 27.1 tokens |
| API cost | $0.000887 | $0.000580 |

Different test projects explain the difference; this was not further training.
The new test also changed execution software and input permissions, so its
results are research comparisons, not approved customer quotes.

Thirteen broader bricks, tools, industry variations and reliable prediction
ranges still need measurement and testing.

See the [HTML report](customer-model-report.html) for charts, source links
and limitations.
