.PHONY: all quick test figures clean
all:            ## full analysis, figures and document
	python run_all.py
quick:          ## coarse profile grid, smoke test
	python run_all.py --quick
test:
	python -m pytest tests -q
clean:
	rm -rf results/results.json results/figures/*.png results/*.pdf \
	       results/run.log **/__pycache__
