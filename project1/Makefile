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
	rm -f figs/r2_vs_degree_ols.pdf
	rm -f figs/mse_vs_degree_ols.pdf
	rm -f figs/theta1_vs_degree_ols.pdf
	rm -f figs/mse_vs_degree_multiple_samples_ols.pdf


	rm -f figs/mse_vs_degree_lambdas_N*.pdf
	rm -f figs/r2_vs_degree_lambdas_N*.pdf
	rm -f figs/theta1_evolution_lambdas_N*.pdf


help:
	@echo "Available targets:"
	@echo "  ols    - Run OLS analysis"
	@echo "  ridge  - Run Ridge regression analysis"
	@echo "  gd     - Run Gradient Descent analysis"
	@echo "  all    - Run both analyses"
	@echo "  clean  - Remove generated figures"
	@echo "  help   - Show this help message"