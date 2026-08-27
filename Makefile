.PHONY: help test demo prove l1 l2 l5 hook dashboard scope serve corpus clean

help:
	@echo "OpenHarness — targets:"
	@echo "  make test       run the unit + benchmark test suite"
	@echo "  make demo       replay the demo sessions, write dashboard.html"
	@echo "  make prove      run L1 + L2 benchmarks and the L3 hook self-test"
	@echo "  make l1         L1 conformance-detection benchmark"
	@echo "  make l2         L2 enforcement ablation"
	@echo "  make l5         L5 precedence & conflict ablation"
	@echo "  make hook       L3 live-hook self-test"
	@echo "  make dashboard  build the harness-cards dashboard"
	@echo "  make scope      Project Yield demo: tokens + value + impact"
	@echo "  make serve      run the Project Yield web prototype"
	@echo "  make corpus     regenerate the synthetic engagement corpus"
	@echo "  make clean      remove generated artifacts"

test:
	python -m pytest -q

demo:
	python -m examples.demo_session

l1:
	python -m benchmark.l1_conformance

l2:
	python -m benchmark.l2_ablation

l5:
	python -m precedence.experiment

agt:
	python -m precedence.agt_demo

hook:
	python3 integrations/claude_code_hook.py --selftest

# The full proof package: measure (L1), improve (L2), plug onto an agent (L3),
# and fix ordering/conflict failures (L5).
prove: l1 l2 l5 hook
	@echo ""
	@echo "✓ proof complete — see benchmark/reports/ and precedence/reports/"

dashboard:
	python -m openharness.cli dashboard -o dashboard.html

# Project Yield — the scoping prototype built on the token model.
scope:
	python -m examples.project_yield_demo

serve:
	python -m project_yield serve --open

# Deterministic and seeded: this must reproduce experiments/engagements.jsonl
# byte for byte, and a test asserts that it does.
corpus:
	python -m experiments.make_engagements > experiments/engagements.jsonl

clean:
	rm -f dashboard.html session_dashboard.html
	rm -rf .openharness __pycache__ */__pycache__ */*/__pycache__ .pytest_cache *.egg-info
