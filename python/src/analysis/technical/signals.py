"""TechnicalAnalysisAgent — top-level interface for the technical analysis pipeline.

Orchestrates indicators, feature engineering, all ML models, ensemble,
training, and evaluation into a single coherent API.

Signal gating:
- Minimum 75% confidence threshold
- Minimum 4 confluences for any trade signal (tiered confidence at 5/6+)
- Kill zone adjustment: +15% confidence in kill zones, -15% outside
- Generates detailed reasoning for every signal
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.types import Signal, Direction, MarketData
from src.analysis.technical.indicators import (
    compute_indicators,
    compute_rule_signals,
    is_in_kill_zone,
)
from src.analysis.technical.feature_engineering import FeatureEngineer
from src.analysis.technical.lstm_model import LSTMModel
from src.analysis.technical.transformer_model import TransformerModel
from src.analysis.technical.pattern_cnn import PatternCNN
from src.analysis.technical.ensemble import (
    EnsembleMetaLearner,
    SIGNAL_TIER_C,
    generate_reasoning,
)
from src.analysis.technical.training import WalkForwardTrainer
from src.analysis.technical.evaluation import ModelEvaluator, BacktestEngine, ModelMetrics

logger = logging.getLogger(__name__)

# Signal gating constants
MIN_CONFIDENCE_THRESHOLD = 75.0
MIN_CONFLUENCES = 4

# Default config path
DEFAULT_CONFIG_PATH = "src/config/default.yaml"


class TechnicalAnalysisAgent:
    """Full technical analysis pipeline agent.

    Provides three main interfaces:
        - analyze(asset, data) -> Signal: Run the complete pipeline
        - train(asset, data): Retrain all models
        - get_model_performance() -> dict: Model health check
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_override: Optional[Dict] = None,
        checkpoint_dir: str = "checkpoints/technical",
    ):
        """Initialize the agent with configuration.

        Args:
            config_path: Path to YAML config file.
            config_override: Dict to override specific config values.
            checkpoint_dir: Directory for model checkpoints.
        """
        self.config = self._load_config(config_path, config_override)
        self.checkpoint_dir = checkpoint_dir

        tech_config = self.config.get("technical", {})
        lstm_cfg = tech_config.get("lstm", {})
        tf_cfg = tech_config.get("transformer", {})

        # Feature engineer (shared)
        self.feature_engineer = FeatureEngineer(
            lookback=lstm_cfg.get("lookback_periods", 60),
            prediction_horizon=8,
            move_threshold=0.005,
        )

        # Models (initialized lazily on first analyze/train call)
        self._input_size: Optional[int] = None
        self._lstm: Optional[LSTMModel] = None
        self._transformer: Optional[TransformerModel] = None
        self._cnn: Optional[PatternCNN] = None
        self._ensemble: Optional[EnsembleMetaLearner] = None
        self._trainer: Optional[WalkForwardTrainer] = None
        self._evaluator = ModelEvaluator()

        # Cache config values
        self._lstm_cfg = lstm_cfg
        self._tf_cfg = tf_cfg

        # State tracking
        self._last_train_time: Optional[datetime] = None
        self._performance_history: List[ModelMetrics] = []
        self._is_initialized = False

    def _load_config(
        self, config_path: Optional[str], override: Optional[Dict]
    ) -> Dict:
        """Load config from YAML file, optionally override values."""
        config = {}
        path = config_path or DEFAULT_CONFIG_PATH
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")

        if override:
            config = _deep_merge(config, override)

        return config

    def _ensure_initialized(self, n_features: int) -> None:
        """Lazily initialize models once we know the feature count."""
        if self._is_initialized and self._input_size == n_features:
            return

        self._input_size = n_features
        lstm_cfg = self._lstm_cfg
        tf_cfg = self._tf_cfg

        self._lstm = LSTMModel(
            input_size=n_features,
            hidden_size=lstm_cfg.get("hidden_size", 128),
            num_layers=lstm_cfg.get("num_layers", 3),
            dropout=lstm_cfg.get("dropout", 0.3),
            learning_rate=lstm_cfg.get("learning_rate", 0.001),
            epochs=lstm_cfg.get("epochs", 100),
            batch_size=lstm_cfg.get("batch_size", 64),
        )

        self._transformer = TransformerModel(
            input_size=n_features,
            d_model=tf_cfg.get("d_model", 128),
            nhead=tf_cfg.get("nhead", 8),
            num_layers=tf_cfg.get("num_layers", 4),
            dropout=tf_cfg.get("dropout", 0.1),
        )

        self._cnn = PatternCNN(
            input_channels=5,
            seq_length=self.feature_engineer.lookback,
        )

        self._ensemble = EnsembleMetaLearner()

        self._trainer = WalkForwardTrainer(
            feature_engineer=self.feature_engineer,
            lstm_model=self._lstm,
            transformer_model=self._transformer,
            pattern_cnn=self._cnn,
            ensemble=self._ensemble,
            checkpoint_dir=self.checkpoint_dir,
        )

        self._is_initialized = True
        logger.info(f"TechnicalAnalysisAgent initialized with {n_features} features")

    def analyze(
        self,
        asset: str,
        data: pd.DataFrame,
        timeframe: str = "1h",
    ) -> Signal:
        """Run the full technical analysis pipeline on latest data.

        Args:
            asset: Asset identifier (e.g., "BTC/USDT").
            data: OHLCV DataFrame with DatetimeIndex.
            timeframe: Data timeframe.

        Returns:
            Unified ensemble Signal with direction and confidence.
        """
        if len(data) < 100:
            logger.warning(f"Insufficient data for {asset}: {len(data)} bars")
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=0.0,
                source="technical",
                timeframe=timeframe,
                metadata={"reason": "insufficient_data"},
            )

        # Compute indicators and features
        df = compute_indicators(data)
        df = compute_rule_signals(df)
        df_feat = self.feature_engineer.build_features(df)

        X, feat_cols = self.feature_engineer.get_feature_matrix(df_feat, fit_scaler=True)
        self._ensure_initialized(len(feat_cols))

        lookback = self.feature_engineer.lookback
        if len(X) <= lookback:
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=0.0,
                source="technical",
                timeframe=timeframe,
            )

        # Get the most recent sequence for LSTM/Transformer
        latest_seq = X[-lookback:]  # (lookback, features)

        # LSTM prediction
        lstm_signal = self._lstm.predict(latest_seq, asset=asset, timeframe=timeframe)

        # Transformer prediction
        tf_signal = self._transformer.predict(latest_seq, asset=asset, timeframe=timeframe)

        # CNN pattern detection (needs raw OHLCV)
        ohlcv = df_feat[["open", "high", "low", "close", "volume"]].values[-lookback:]
        ohlcv_input = self._cnn.prepare_input(ohlcv[np.newaxis, :])
        cnn_signal = self._cnn.predict(ohlcv_input, asset=asset, timeframe=timeframe)

        # Rule-based signal
        rule_val = 0.0
        if "rule_signal" in df_feat.columns:
            rule_val = float(df_feat["rule_signal"].iloc[-1])

        # Indicator features for ensemble
        indicator_feats = {}
        for col in ["rsi", "macd_hist", "bb_pct", "adx", "volume_ratio"]:
            if col in df_feat.columns:
                val = df_feat[col].iloc[-1]
                indicator_feats[col] = float(val) if np.isfinite(val) else 0.0

        # Ensemble prediction
        ensemble_signal = self._ensemble.predict(
            lstm_signal=lstm_signal,
            transformer_signal=tf_signal,
            cnn_signal=cnn_signal,
            rule_signal=rule_val,
            indicator_features=indicator_feats,
            asset=asset,
            timeframe=timeframe,
        )

        # Enrich metadata
        ensemble_signal.metadata["indicators"] = {
            "rsi": indicator_feats.get("rsi"),
            "macd_hist": indicator_feats.get("macd_hist"),
            "bb_pct": indicator_feats.get("bb_pct"),
            "adx": indicator_feats.get("adx"),
            "atr": float(df["atr"].iloc[-1]) if "atr" in df.columns else None,
        }

        # ── Signal gating ──────────────────────────────────────────
        ensemble_signal = self._apply_signal_gates(ensemble_signal, df)

        return ensemble_signal

    def _apply_signal_gates(
        self,
        signal: Signal,
        df: pd.DataFrame,
    ) -> Signal:
        """Apply signal gating rules to filter low-quality signals.

        Gates:
        1. Minimum confidence threshold (75%)
        2. Minimum 4 confluences (with tiered confidence boost at 5/6+)
        3. Kill zone confidence adjustment (+15% in / -15% out)

        Args:
            signal: The raw ensemble signal.
            df: DataFrame with indicator columns (for kill zone check).

        Returns:
            Gated signal — downgraded to HOLD if gates are not passed.
        """
        gate_reasons = []

        # Gate 1: Minimum confidence
        if signal.confidence < MIN_CONFIDENCE_THRESHOLD:
            gate_reasons.append(
                f"Confidence {signal.confidence:.1f}% below threshold {MIN_CONFIDENCE_THRESHOLD}%"
            )

        # Gate 2: Minimum confluences with tiered confidence adjustment
        confluences = signal.metadata.get("confluences", 0)
        if confluences < MIN_CONFLUENCES:
            gate_reasons.append(
                f"Only {confluences} confluences (minimum {MIN_CONFLUENCES} required)"
            )
        elif confluences == 5:
            signal.confidence *= 1.05  # +5% for 5 confluences
        elif confluences >= 6:
            signal.confidence *= 1.10  # +10% for 6+ confluences

        # Gate 3: Kill zone confidence adjustment (boost/reduce, never reject)
        kill_zone_series = is_in_kill_zone(df)
        if len(kill_zone_series) > 0:
            in_kill_zone = kill_zone_series.iloc[-1] > 0
        else:
            in_kill_zone = False

        if in_kill_zone:
            signal.confidence *= 1.15  # +15% boost during high institutional volume
        else:
            signal.confidence *= 0.85  # -15% reduction outside kill zones

        signal.metadata["in_kill_zone"] = in_kill_zone

        # Apply gates: downgrade to HOLD if any gate fails
        if gate_reasons and signal.direction != Direction.HOLD:
            signal.metadata["gate_filtered"] = True
            signal.metadata["gate_reasons"] = gate_reasons
            signal.direction = Direction.HOLD
            logger.info(
                f"Signal gated to HOLD: {'; '.join(gate_reasons)}"
            )
        else:
            signal.metadata["gate_filtered"] = False

        return signal

    def train(
        self,
        asset: str,
        data: pd.DataFrame,
        epochs: int = 50,
        walk_forward: bool = False,
    ) -> Dict[str, Any]:
        """Train or retrain all models on provided data.

        Args:
            asset: Asset identifier.
            data: OHLCV DataFrame.
            epochs: Training epochs per model.
            walk_forward: If True, run full walk-forward validation.
                If False, simple train on latest data.

        Returns:
            Training metrics dict.
        """
        logger.info(f"Training technical models for {asset} on {len(data)} bars")

        # Compute features to determine input size
        df = compute_indicators(data)
        df = compute_rule_signals(df)
        df_feat = self.feature_engineer.build_features(df)
        X, feat_cols = self.feature_engineer.get_feature_matrix(df_feat, fit_scaler=True)

        self._ensure_initialized(len(feat_cols))

        if walk_forward:
            result = self._trainer.run_walk_forward(data, epochs_per_fold=epochs)
        else:
            result = self._trainer.retrain_on_latest(data, epochs=epochs)

        self._last_train_time = datetime.utcnow()
        logger.info(f"Training complete for {asset}")
        return result

    def get_model_performance(self) -> Dict[str, Any]:
        """Get a health check of all models.

        Returns:
            Dict with model status, weights, and recent accuracy.
        """
        result = {
            "initialized": self._is_initialized,
            "last_train_time": self._last_train_time.isoformat() if self._last_train_time else None,
            "needs_retrain": self._trainer.needs_retrain() if self._trainer else True,
        }

        if self._is_initialized:
            result["models"] = {
                "lstm": {
                    "trained": self._lstm.is_trained,
                    "train_losses": len(self._lstm.train_losses),
                    "final_loss": self._lstm.train_losses[-1] if self._lstm.train_losses else None,
                },
                "transformer": {
                    "trained": self._transformer.is_trained,
                    "train_losses": len(self._transformer.train_losses),
                    "final_loss": self._transformer.train_losses[-1] if self._transformer.train_losses else None,
                },
                "pattern_cnn": {
                    "trained": self._cnn.is_trained,
                },
                "ensemble": {
                    "trained": self._ensemble.is_trained,
                    "model_weights": self._ensemble.get_model_weights(),
                    "model_accuracy": self._ensemble.get_model_accuracy(),
                    "rolling_accuracy": self._ensemble.get_rolling_accuracy(),
                },
            }
            if self._trainer:
                result["degradation_flags"] = self._trainer.get_degradation_flags()

        return result

    def save_models(self, prefix: str = "latest") -> None:
        """Save all model checkpoints."""
        if not self._is_initialized:
            logger.warning("Models not initialized, nothing to save")
            return

        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._lstm.save(os.path.join(self.checkpoint_dir, f"{prefix}_lstm.pt"))
        self._transformer.save(os.path.join(self.checkpoint_dir, f"{prefix}_transformer.pt"))
        self._cnn.save(os.path.join(self.checkpoint_dir, f"{prefix}_cnn.pt"))
        logger.info(f"All models saved with prefix '{prefix}'")

    def load_models(self, prefix: str = "latest") -> None:
        """Load all model checkpoints.

        Requires models to be initialized first (call analyze or train once).
        """
        if not self._is_initialized:
            logger.warning("Models not initialized. Call analyze() or train() first.")
            return

        import os
        lstm_path = os.path.join(self.checkpoint_dir, f"{prefix}_lstm.pt")
        tf_path = os.path.join(self.checkpoint_dir, f"{prefix}_transformer.pt")
        cnn_path = os.path.join(self.checkpoint_dir, f"{prefix}_cnn.pt")

        if os.path.exists(lstm_path):
            self._lstm.load(lstm_path)
        if os.path.exists(tf_path):
            self._transformer.load(tf_path)
        if os.path.exists(cnn_path):
            self._cnn.load(cnn_path)

        logger.info(f"Models loaded with prefix '{prefix}'")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
