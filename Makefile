SHELL := /bin/bash

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	python3 -m pip install --user --upgrade --break-system-packages virtualenv==20.29.1
	python3 -m virtualenv $(VENV)
	$(PIP) install --upgrade pip==26.0.1
	bash scripts/install_torch.sh "$(PIP)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PY) -m srls.data.prepare --config $(CONFIG)
	$(PY) -m srls.data.verify --config $(CONFIG)

train:
	$(PY) -m srls.train --config $(CONFIG) --run baseline
	$(PY) -m srls.train --config $(CONFIG) --run main
	$(PY) -m srls.train --config $(CONFIG) --run ablation_no_routing

eval:
	$(PY) -m srls.eval --config $(CONFIG) --run baseline
	$(PY) -m srls.eval --config $(CONFIG) --run main
	$(PY) -m srls.eval --config $(CONFIG) --run ablation_no_routing

report:
	$(PY) -m srls.report --config $(CONFIG)

all: setup data train eval report

clean:
	rm -rf $(VENV) data/processed outputs report.md
