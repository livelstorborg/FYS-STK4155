.PHONY: clean part_d_clean part_b_clean part_c_clean part_e_clean part_f_clean all_clean help

# Default target - show help
help:
	@echo "Available targets:"
	@echo "  make clean         - Remove all PDF files from code/notebooks/figs/"
	@echo "  make part_d_clean  - Remove Part D PDF files from code/notebooks/figs/"
	@echo "  make part_b_clean  - Remove Part B PDF files from code/notebooks/figs/"
	@echo "  make part_c_clean  - Remove Part C PDF files from code/notebooks/figs/"
	@echo "  make part_e_clean  - Remove Part E PDF files from code/notebooks/figs/"
	@echo "  make part_f_clean  - Remove Part F PDF files from code/notebooks/figs/"
	@echo "  make all_clean     - Remove all PDFs and CSV results"

# Clean all PDF files in figs folder
clean:
	@echo "Removing all PDF files from code/notebooks/figs/..."
	@rm -f code/notebooks/figs/*.pdf
	@echo "✓ Done"

# Clean Part D specific files
part_d_clean:
	@echo "Removing Part D PDF files..."
	@rm -f code/notebooks/figs/bias_variance_sigmoid[0-9].pdf
	@rm -f code/notebooks/figs/bias_variance_relu[0-9].pdf
	@rm -f code/notebooks/figs/bias_variance_leakyrelu[0-9].pdf
	@rm -f code/notebooks/figs/mse_heatmap_*.pdf
	@rm -f code/notebooks/figs/*_layers_comparison.pdf
	@rm -f code/notebooks/figs/heatmap_*.pdf
	@rm -f code/notebooks/figs/learning_curves_*.pdf
	@echo "✓ Part D PDFs removed"

# Clean Part B specific files (adjust patterns as needed)
part_b_clean:
	@echo "Removing Part B PDF files..."
	@rm -f code/notebooks/figs/*_gd_*.pdf
	@rm -f code/notebooks/figs/*_adam_*.pdf
	@rm -f code/notebooks/figs/*_rms*.pdf
	@echo "✓ Part B PDFs removed"

# Clean Part C specific files (adjust patterns as needed)
part_c_clean:
	@echo "Removing Part C PDF files..."
	@rm -f code/notebooks/figs/complexity_*.pdf
	@echo "✓ Part C PDFs removed"

# Clean Part E specific files (adjust patterns as needed)
part_e_clean:
	@echo "Removing Part E PDF files..."
	@rm -f code/notebooks/figs/*_l1_*.pdf
	@rm -f code/notebooks/figs/*_l2_*.pdf
	@echo "✓ Part E PDFs removed"

# Clean Part F specific files (adjust patterns as needed)
part_f_clean:
	@echo "Removing Part F PDF files..."
	@rm -f code/notebooks/figs/*_classification_*.pdf
	@echo "✓ Part F PDFs removed"

# Clean everything (PDFs and results)
all_clean: clean
	@echo "Removing CSV result files..."
	@rm -f code/notebooks/results/*.csv
	@rm -f code/notebooks/results/*.json
	@echo "✓ All files cleaned"
