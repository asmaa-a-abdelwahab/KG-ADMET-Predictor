"""Backward-compatible entry point for the default PRING modeling baseline."""
from .stage1_tabular import build_parser, main, run

__all__ = ["build_parser", "main", "run"]
