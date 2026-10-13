"""Exact one-dimensional consolidation of layered soil with certified error bounds."""
__version__ = "0.1.0"
from .core import Column, Interval, Layer  # noqa: E402
from .model import Case, LoadStage, case_from_dict, load_case, secant_mv  # noqa: E402

__all__ = ["Column", "Interval", "Layer", "Case", "LoadStage", "case_from_dict", "load_case", "secant_mv"]
