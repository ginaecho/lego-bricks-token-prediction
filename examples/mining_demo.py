"""Mine real repositories, classify the work, and see what we cannot price yet.

Run with: python -m examples.mining_demo

Probes tell you what a unit of work costs. Mining tells you what work a project
is actually made of. Crossing the two produces a measurement backlog: the kinds
of real work you have never measured, ranked by how much of the project they
would unlock.
"""

from token_yield.learn import seeded_store
from token_yield.mine import classify, coverage, distribution, mine_repo
from token_yield.plan import PlanForecaster, WorkPlan

REPOS = [
    ("/home/user/psf/requests", "requests"),
    ("/home/user/pallets/click", "click"),
    ("/home/user/harness-dose", "harness-dose"),
]


def rule(title: str) -> None:
    print()
    print(title)
    print("=" * 74)


def main() -> None:
    store = seeded_store()

    rule("1. MINE — what is the real work made of?")
    mined = []
    for path, name in REPOS:
        tasks = mine_repo(path, limit=200, repo=name)
        mined += tasks
        print(f"  {name:<16} {len(tasks):>4} commits")
    if not mined:
        print("\n  No repositories available to mine in this environment.")
        return

    print()
    print(f"  {'kind':<16}{'n':>5}{'share':>8}{'med lines':>11}"
          f"{'med files':>11}{'confidence':>12}")
    for kind, d in distribution(mined).items():
        print(f"  {kind:<16}{d['count']:>5}{d['share']:>8.0%}"
              f"{d['median_lines']:>11}{d['median_files']:>11}"
              f"{d['mean_confidence']:>12.2f}")

    rule("2. CLASSIFY — every call shows the evidence that fired it")
    for t in mined[:6]:
        print(f"  {t.sha[:8]}  {t.kind:<13} conf {t.confidence:.2f}  — {t.why}")
    print("  ...")
    low = [t for t in mined if t.confidence <= 0.3]
    print(f"\n  {len(low)} of {len(mined)} commits matched no strong rule and are")
    print(f"  labelled 'code_change' at low confidence rather than guessed into")
    print(f"  a category they might not belong to.")

    rule("3. COVERAGE — what can we actually price?")
    cov = coverage(mined, store.kinds())
    print(cov.summary())

    rule("4. WHAT THAT MEANS FOR A BUDGET")
    print("  Kinds we HAVE measured, and what they cost:")
    for kind in store.kinds():
        m = store.model_for(kind)
        print(f"    {kind:<16} {m.equation()}")

    print()
    print("  Pricing a plan that includes an unmeasured kind:")
    plan = (WorkPlan("next sprint")
            .add("docs", 4, count=3)
            .add("bug_fix", 2, count=5))          # never probed
    fc = PlanForecaster(store).forecast(plan)
    print()
    for line in fc.summary().splitlines():
        print("    " + line)
    print()
    print("  The budget is not wrong — it is incomplete, and it says so.")
    print("  That is the difference between a forecast and a guess.")


if __name__ == "__main__":
    main()
