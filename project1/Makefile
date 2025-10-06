PYTHON = python3

.PHONY: all clean ols ridge gd help

.DEFAULT_GOAL := help

all: ols ridge gd

ols:
	$(PYTHON) main_ols.py

ridge:
	$(PYTHON) main_ridge.py

gd:
	$(PYTHON) main_gd.py

clean:
	rm -rf figs/*
	rm -rf figs/


help:
	@echo "Available targets:"
	@echo "  ols    - Run OLS analysis"
	@echo "  ridge  - Run Ridge regression analysis"
	@echo "  gd     - Run Gradient Descent analysis"
	@echo "  all    - Run both analyses"
	@echo "  clean  - Remove generated figures"
	@echo "  help   - Show this help message"