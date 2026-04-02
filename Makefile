PYTHON ?= python3

.PHONY: install ingest index ask eval test

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest

index:
	$(PYTHON) -m src.index

ask:
	$(PYTHON) -m src.main --question "$(Q)"

eval:
	$(PYTHON) -m src.eval

test:
	$(PYTHON) -m pytest -q
