.PHONY: browse sync run cluster help

help:
	@echo "Available commands:"
	@echo "  make browse  - Browse the Chroma database collection"
	@echo "  make sync    - Sync dependencies with uv"
	@echo "  make run     - Run the main PDF processing script"
	@echo "  make cluster - Run the PDF clustering script"

browse:
	uv run chroma browse pdf_cluster_data --local --path ./chroma_db

sync:
	uv sync

run:
	uv run python main.py

cluster:
	uv run python cluster_pdfs.py
