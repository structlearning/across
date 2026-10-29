PY ?= python

.PHONY: help install download eval train tables clean distclean

help:
	@echo "  make install     Install pinned dependencies"
	@echo "  make download    Pull data + checkpoints from HF Hub"
	@echo "  make eval        Reproduce the paper's tables"
	@echo "  make tables      Re-render tables from existing testMAPs/"
	@echo "  make all         install + download + eval"
	@echo "  make train       Full retrain from scratch (see docs/retraining.md)"
	@echo "  make clean       Remove testMAPs/ and rendered tables"
	@echo "  make distclean   Also remove downloaded data and checkpoints"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

download:
	$(PY) -B scripts/reproduce/download_assets.py

eval:
	$(PY) -B scripts/reproduce/eval_main_tables.py

tables:
	mkdir -p output/tables
	$(PY) -B scripts/latex/make_results_table.py  --out output/tables/results_table.tex
	$(PY) -B scripts/latex/make_results_table.py  --out output/tables/results_table.tex  --appendix
	$(PY) -B scripts/latex/make_augnet_table.py   --out output/tables/augnet_table.tex
	$(PY) -B scripts/latex/make_augnet_table.py   --out output/tables/augnet_table.tex   --appendix
	$(PY) -B scripts/latex/make_fmap_table.py     --out output/tables/fmap_table.tex
	$(PY) -B scripts/latex/make_fmap_table.py     --out output/tables/fmap_table.tex     --appendix
	$(PY) -B scripts/latex/make_ablation_table.py --out output/tables/ablation_table.tex
	$(PY) -B scripts/latex/make_ablation_table.py --out output/tables/ablation_table.tex --appendix

all: install download eval

train:
	@echo "See docs/retraining.md for the retrain-from-scratch pipeline."
	@exit 1

clean:
	rm -rf testMAPs/ output/tables/

distclean: clean
	rm -rf data/ logs/ latestModels/
