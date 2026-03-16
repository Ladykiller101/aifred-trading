"""Technical Analysis module.

Provides the TechnicalAnalysisAgent as the main entry point for
running the full technical analysis pipeline (indicators, ML models,
ensemble signal generation, training, and evaluation).
"""

from src.analysis.technical.signals import TechnicalAnalysisAgent

__all__ = ["TechnicalAnalysisAgent"]
