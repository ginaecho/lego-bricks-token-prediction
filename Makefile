.PHONY: help test demo prove l1 l2 l5 hook dashboard studio assay-audit assay-corpus clean

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
	@echo ""
	@echo "Token-yield assay (needs Python >= 3.11):"
	@echo "  make studio       run the visual demo in a browser"
	@echo "  make assay-audit  replay the reference table beside its variable-portion rescore"
	@echo "  make assay-corpus regenerate the demo corpus"
	@echo ""
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

studio:
	python -m assay studio

assay-audit:
	python -m assay audit

assay-corpus:
	python -m demo.generate_corpus

clean:
	rm -f dashboard.html session_dashboard.html
	rm -rf .openharness __pycache__ */__pycache__ */*/__pycache__ .pytest_cache *.egg-info work
