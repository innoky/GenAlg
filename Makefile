PY    ?= .venv/bin/python
SEEDS ?= 0-19
JOBS  ?= $(shell nproc)

.PHONY: all venv test lab1 lab2 lab3 reports clean

all: test lab1 lab2 lab3 reports

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lab1:
	$(PY) -m lab1.run_experiments --config lab1/configs/*.json --seeds $(SEEDS) --out lab1/results

lab2:
	$(PY) -m lab2.data_gen --seed 527 --out lab2/data/instance.json
	$(PY) -m lab2.run_experiments --config lab2/configs/*.json --seeds $(SEEDS) --out lab2/results
	$(PY) -m lab2.report_tables --out lab2/results

lab3:
	$(PY) -m lab3.data_gen --seed 327 --out lab3/data
	$(PY) -m lab3.run_experiments --config lab3/configs/*.json --seeds $(SEEDS) --jobs $(JOBS) --out lab3/results

# подставить свежие таблицы из results/ в отчёты
reports:
	$(PY) -m eaga.report_include lab1/report.md lab2/report.md lab3/report.md

clean:
	rm -rf lab1/results/* lab2/results/* lab3/results/*
