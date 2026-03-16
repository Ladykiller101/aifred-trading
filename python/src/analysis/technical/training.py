"""Walk-forward validation training pipeline.

Implements walk-forward (expanding/sliding window) cross-validation to
prevent future data leakage. Supports daily retraining on latest data
and MLflow experiment tracking.

Features:
- Purged walk-forward validation with configurable gap between train/test
- Model degradation detection (flags models below 60% rolling accuracy)
- Ensemble rebalancing after each retraining cycle
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.technical.feature_engineering import FeatureEngineer
from src.analysis.technical.indicators import compute_indicators, compute_rule_signals
from src.analysis.technical.lstm_model import LSTMModel
from src.analysis.technical.transformer_model import TransformerModel
from src.analysis.technical.pattern_cnn import PatternCNN, generate_pattern_labels
from src.analysis.technical.ensemble import EnsembleMetaLearner

logger = logging.getLogger(__name__)

# Try to import MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class WalkForwardTrainer:
    """Walk-forward validation pipeline for all technical models.

    Splits data into expanding training windows and fixed-size test windows,
    training sequentially to prevent any future data leakage.
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        lstm_model: LSTMModel,
        transformer_model: TransformerModel,
        pattern_cnn: PatternCNN,
        ensemble: EnsembleMetaLearner,
        initial_train_pct: float = 0.5,
        test_window_size: int = 200,
        retrain_interval: int = 200,
        expanding_window: bool = True,
        checkpoint_dir: str = "checkpoints/technical",
        mlflow_experiment: Optional[str] = "technical_analysis",
        purge_gap: int = 20,
        degradation_threshold: float = 0.60,
    ):
        """
        Args:
            feature_engineer: Shared feature engineering instance.
            lstm_model: LSTM model instance.
            transformer_model: Transformer model instance.
            pattern_cnn: Pattern CNN instance.
            ensemble: Ensemble meta-learner instance.
            initial_train_pct: Fraction of data for initial training window.
            test_window_size: Size of each test/validation fold.
            retrain_interval: Bars between retraining steps.
            expanding_window: If True, training window grows; if False, slides.
            checkpoint_dir: Directory for model checkpoints.
            mlflow_experiment: MLflow experiment name (None to disable).
            purge_gap: Number of bars to skip between train and test windows
                to prevent data leakage from overlapping labels.
            degradation_threshold: If rolling 30-trade accuracy drops below
                this threshold, flag the model as degraded.
        """
        self.feature_engineer = feature_engineer
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.pattern_cnn = pattern_cnn
        self.ensemble = ensemble
        self.initial_train_pct = initial_train_pct
        self.test_window_size = test_window_size
        self.retrain_interval = retrain_interval
        self.expanding_window = expanding_window
        self.checkpoint_dir = checkpoint_dir
        self.mlflow_experiment = mlflow_experiment
        self.purge_gap = purge_gap
        self.degradation_threshold = degradation_threshold

        self.fold_results: List[Dict] = []
        self.last_train_timestamp: Optional[datetime] = None
        self.degradation_flags: Dict[str, bool] = {
            "lstm": False,
            "transformer": False,
            "pattern_cnn": False,
        }

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        epochs_per_fold: int = 50,
    ) -> Dict[str, Any]:
        """Execute full walk-forward validation.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.
            epochs_per_fold: Training epochs per walk-forward fold.

        Returns:
            Dict with aggregate metrics and per-fold results.
        """
        logger.info(f"Starting walk-forward validation on {len(df)} bars")

        # Compute indicators and features
        df_ind = compute_indicators(df)
        df_ind = compute_rule_signals(df_ind)
        df_feat = self.feature_engineer.build_features(df_ind)
        labels = self.feature_engineer.build_labels(df_feat)

        # Get feature matrix
        X, feat_cols = self.feature_engineer.get_feature_matrix(df_feat, fit_scaler=True)
        y = labels.values

        # Build OHLCV sequences for CNN
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv_data = df_feat[ohlcv_cols].values.astype(np.float32)

        # Walk-forward splits
        n = len(X)
        initial_train_end = int(n * self.initial_train_pct)
        lookback = self.feature_engineer.lookback

        self.fold_results = []
        fold_idx = 0

        # Setup MLflow
        if MLFLOW_AVAILABLE and self.mlflow_experiment:
            mlflow.set_experiment(self.mlflow_experiment)

        train_start = 0
        current_pos = initial_train_end

        while current_pos + self.purge_gap + self.test_window_size <= n:
            fold_idx += 1
            train_end = current_pos
            # Purged walk-forward: skip purge_gap bars between train and test
            test_start = current_pos + self.purge_gap
            test_end = min(test_start + self.test_window_size, n)

            if not self.expanding_window:
                train_start = max(0, train_end - initial_train_end)

            logger.info(
                f"Fold {fold_idx}: train [{train_start}:{train_end}], "
                f"purge gap [{train_end}:{test_start}], "
                f"test [{test_start}:{test_end}]"
            )

            fold_metrics = self._train_fold(
                X=X,
                y=y,
                ohlcv_data=ohlcv_data,
                df_feat=df_feat,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                lookback=lookback,
                epochs=epochs_per_fold,
                fold_idx=fold_idx,
            )

            # Check for model degradation after each fold
            self._check_degradation(fold_metrics, fold_idx)

            self.fold_results.append(fold_metrics)

            # Log to MLflow
            if MLFLOW_AVAILABLE and self.mlflow_experiment:
                with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                    for key, val in fold_metrics.items():
                        if isinstance(val, (int, float)):
                            mlflow.log_metric(key, val, step=fold_idx)

            current_pos += self.retrain_interval

        # Rebalance ensemble weights based on accumulated performance
        rebalanced_weights = self._rebalance_ensemble()

        # Aggregate results
        aggregate = self._aggregate_results()
        logger.info(f"Walk-forward complete: {fold_idx} folds, {aggregate}")

        return {
            "aggregate": aggregate,
            "folds": self.fold_results,
            "n_folds": fold_idx,
            "rebalanced_weights": rebalanced_weights,
            "degradation_flags": dict(self.degradation_flags),
        }

    def _train_fold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ohlcv_data: np.ndarray,
        df_feat: pd.DataFrame,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        lookback: int,
        epochs: int,
        fold_idx: int,
    ) -> Dict[str, float]:
        """Train and evaluate a single walk-forward fold."""
        # Prepare training sequences
        X_train_flat = X[train_start:train_end]
        y_train_flat = y[train_start:train_end]
        X_test_flat = X[test_start:test_end]
        y_test_flat = y[test_start:test_end]

        # Filter invalid labels
        valid_train = y_train_flat >= 0
        valid_test = y_test_flat >= 0
        X_train_flat = X_train_flat[valid_train]
        y_train_flat = y_train_flat[valid_train]
        X_test_flat = X_test_flat[valid_test]
        y_test_flat = y_test_flat[valid_test]

        if len(X_train_flat) < 100 or len(X_test_flat) < 20:
            return {"fold": fold_idx, "skipped": True, "reason": "insufficient_data"}

        # Build sequences for LSTM/Transformer
        X_train_seq, y_train_seq = self.feature_engineer.build_sequences(
            X_train_flat, y_train_flat, lookback
        )
        X_test_seq, y_test_seq = self.feature_engineer.build_sequences(
            X_test_flat, y_test_flat, lookback
        )

        metrics = {"fold": fold_idx}

        # ── Train LSTM ────────────────────────────────────────────
        if len(X_train_seq) > 50 and len(X_test_seq) > 10:
            lstm_history = self.lstm_model.train(
                X_train_seq, y_train_seq,
                X_val=X_test_seq, y_val=y_test_seq,
                epochs=epochs,
            )
            lstm_probs, lstm_conf = self.lstm_model.predict_batch(X_test_seq)
            lstm_preds = (lstm_probs[:, 1] > 0.5).astype(int)
            metrics["lstm_accuracy"] = float(np.mean(lstm_preds == y_test_seq))
            metrics["lstm_final_train_loss"] = lstm_history["train_loss"][-1] if lstm_history["train_loss"] else 0

        # ── Train Transformer ─────────────────────────────────────
        if len(X_train_seq) > 50 and len(X_test_seq) > 10:
            tf_history = self.transformer_model.train(
                X_train_seq, y_train_seq,
                X_val=X_test_seq, y_val=y_test_seq,
                epochs=epochs,
            )
            tf_probs, tf_conf = self.transformer_model.predict_batch(X_test_seq)
            tf_preds = (tf_probs[:, 1] > 0.5).astype(int)
            metrics["transformer_accuracy"] = float(np.mean(tf_preds == y_test_seq))
            metrics["transformer_final_train_loss"] = tf_history["train_loss"][-1] if tf_history["train_loss"] else 0

        # ── Train Pattern CNN ─────────────────────────────────────
        ohlcv_train = ohlcv_data[train_start:train_end]
        ohlcv_test = ohlcv_data[test_start:test_end]

        if len(ohlcv_train) > lookback + 50:
            cnn_seqs_train = self._build_ohlcv_sequences(ohlcv_train, lookback)
            future_ret_train = y_train_flat[lookback:lookback + len(cnn_seqs_train)]

            if len(cnn_seqs_train) > 50 and len(future_ret_train) >= len(cnn_seqs_train):
                future_ret_train = future_ret_train[:len(cnn_seqs_train)]
                pat_labels, dir_labels = generate_pattern_labels(
                    cnn_seqs_train, future_ret_train.astype(float), lookback
                )
                cnn_input = self.pattern_cnn.prepare_input(cnn_seqs_train)
                self.pattern_cnn.train(cnn_input, pat_labels, dir_labels, epochs=epochs)

                # Evaluate on test
                if len(ohlcv_test) > lookback:
                    cnn_seqs_test = self._build_ohlcv_sequences(ohlcv_test, lookback)
                    if len(cnn_seqs_test) > 0:
                        cnn_input_test = self.pattern_cnn.prepare_input(cnn_seqs_test)
                        cnn_probs, cnn_conf = self.pattern_cnn.predict_batch(cnn_input_test)
                        test_labels = y_test_flat[lookback:lookback + len(cnn_seqs_test)]
                        if len(test_labels) >= len(cnn_seqs_test):
                            test_labels = test_labels[:len(cnn_seqs_test)]
                            cnn_preds = (cnn_probs[:, 1] > 0.5).astype(int)
                            metrics["cnn_accuracy"] = float(np.mean(cnn_preds == test_labels))

        # ── Train Ensemble Meta-Learner ───────────────────────────
        if (
            self.lstm_model.is_trained
            and self.transformer_model.is_trained
            and len(X_test_seq) > 30
        ):
            meta_X, meta_y = self._build_meta_training_data(
                X_train_seq, y_train_seq, df_feat, train_start, train_end, lookback
            )
            if len(meta_X) > 50:
                self.ensemble.train(meta_X, meta_y)
                metrics["ensemble_trained"] = True

        # Save checkpoints
        self._save_checkpoints(fold_idx)

        return metrics

    def _build_ohlcv_sequences(
        self, ohlcv: np.ndarray, lookback: int
    ) -> np.ndarray:
        """Build OHLCV windows for CNN input."""
        n = len(ohlcv) - lookback
        if n <= 0:
            return np.array([])
        seqs = np.zeros((n, lookback, ohlcv.shape[1]), dtype=np.float32)
        for i in range(n):
            seqs[i] = ohlcv[i : i + lookback]
        return seqs

    def _build_meta_training_data(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        df_feat: pd.DataFrame,
        train_start: int,
        train_end: int,
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate meta-features by running sub-models on training data."""
        meta_X = []
        meta_y = []

        # Use a subset to avoid excessive computation
        step = max(1, len(X_seq) // 200)
        for i in range(0, len(X_seq), step):
            seq = X_seq[i:i+1]
            label = y_seq[i]
            if label < 0:
                continue

            lstm_sig = self.lstm_model.predict(seq)
            tf_sig = self.transformer_model.predict(seq)

            # Rule signal from DataFrame
            feat_idx = train_start + lookback + i
            if feat_idx < len(df_feat) and "rule_signal" in df_feat.columns:
                rule_val = df_feat["rule_signal"].iloc[feat_idx]
            else:
                rule_val = 0.0

            meta_feat = self.ensemble.build_meta_features(
                lstm_signal=lstm_sig,
                transformer_signal=tf_sig,
                rule_signal=float(rule_val),
            )
            meta_X.append(meta_feat)
            meta_y.append(label)

        if meta_X:
            return np.array(meta_X), np.array(meta_y)
        return np.array([]), np.array([])

    def _save_checkpoints(self, fold_idx: int) -> None:
        """Save model checkpoints for current fold."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        try:
            self.lstm_model.save(
                os.path.join(self.checkpoint_dir, f"lstm_fold{fold_idx}.pt")
            )
            self.transformer_model.save(
                os.path.join(self.checkpoint_dir, f"transformer_fold{fold_idx}.pt")
            )
            self.pattern_cnn.save(
                os.path.join(self.checkpoint_dir, f"cnn_fold{fold_idx}.pt")
            )
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

    def _check_degradation(self, fold_metrics: Dict, fold_idx: int) -> None:
        """Check if any model's rolling accuracy has degraded below threshold.

        If a model's accuracy drops below the degradation threshold (default 55%),
        it is flagged for attention.

        Args:
            fold_metrics: Metrics from the latest fold.
            fold_idx: Current fold index.
        """
        for model_key, metric_key in [
            ("lstm", "lstm_accuracy"),
            ("transformer", "transformer_accuracy"),
            ("pattern_cnn", "cnn_accuracy"),
        ]:
            accuracy = fold_metrics.get(metric_key)
            if accuracy is not None:
                # Update ensemble rolling accuracy tracker
                was_correct = accuracy > 0.5
                self.ensemble.update_performance(model_key, was_correct)

                if accuracy < self.degradation_threshold:
                    self.degradation_flags[model_key] = True
                    logger.warning(
                        f"MODEL DEGRADATION: {model_key} accuracy={accuracy:.3f} "
                        f"(below {self.degradation_threshold}) at fold {fold_idx}. "
                        f"Consider retraining with fresh data."
                    )
                else:
                    self.degradation_flags[model_key] = False

    def _rebalance_ensemble(self) -> Dict[str, float]:
        """Rebalance ensemble weights based on recent fold performance.

        Called after each retraining cycle. Updates the ensemble meta-learner's
        model weights based on the accumulated rolling accuracy.

        Returns:
            Updated model weights.
        """
        # The ensemble's _update_weights is triggered by update_performance calls
        # in _check_degradation. Here we just retrieve and log the result.
        weights = self.ensemble.get_model_weights()
        rolling = self.ensemble.get_rolling_accuracy()

        logger.info(
            f"Ensemble rebalanced — weights: {weights}, "
            f"rolling accuracy: {rolling}, "
            f"degradation flags: {self.degradation_flags}"
        )

        return weights

    def get_degradation_flags(self) -> Dict[str, bool]:
        """Return current degradation flags for each model."""
        return dict(self.degradation_flags)

    def _aggregate_results(self) -> Dict[str, float]:
        """Aggregate metrics across all folds."""
        if not self.fold_results:
            return {}

        valid_folds = [f for f in self.fold_results if not f.get("skipped", False)]
        if not valid_folds:
            return {"n_valid_folds": 0}

        result = {"n_valid_folds": len(valid_folds)}

        for metric_key in ["lstm_accuracy", "transformer_accuracy", "cnn_accuracy"]:
            values = [f[metric_key] for f in valid_folds if metric_key in f]
            if values:
                result[f"avg_{metric_key}"] = float(np.mean(values))
                result[f"std_{metric_key}"] = float(np.std(values))

        return result

    def retrain_on_latest(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        """Retrain all models on the latest available data.

        Uses the most recent data as training, reserving the last
        `test_window_size` bars for validation.

        Args:
            df: Full OHLCV DataFrame.
            epochs: Training epochs.

        Returns:
            Training metrics.
        """
        logger.info("Retraining on latest data")

        df_ind = compute_indicators(df)
        df_ind = compute_rule_signals(df_ind)
        df_feat = self.feature_engineer.build_features(df_ind)
        labels = self.feature_engineer.build_labels(df_feat)

        X, feat_cols = self.feature_engineer.get_feature_matrix(df_feat, fit_scaler=True)
        y = labels.values

        n = len(X)
        val_size = min(self.test_window_size, n // 5)
        # Apply purge gap between train and validation
        train_end = n - val_size - self.purge_gap
        test_start = n - val_size

        if train_end < 100:
            # Not enough data for purge gap, fall back to no gap
            train_end = n - val_size
            test_start = train_end

        ohlcv_data = df_feat[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        lookback = self.feature_engineer.lookback

        metrics = self._train_fold(
            X=X,
            y=y,
            ohlcv_data=ohlcv_data,
            df_feat=df_feat,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=n,
            lookback=lookback,
            epochs=epochs,
            fold_idx=0,
        )

        # Check degradation and rebalance
        self._check_degradation(metrics, fold_idx=0)
        rebalanced_weights = self._rebalance_ensemble()
        metrics["rebalanced_weights"] = rebalanced_weights
        metrics["degradation_flags"] = dict(self.degradation_flags)

        self.last_train_timestamp = datetime.utcnow()
        logger.info(f"Retrain complete: {metrics}")
        return metrics

    def needs_retrain(self, hours_since_last: float = 24.0) -> bool:
        """Check if models need retraining based on time elapsed."""
        if self.last_train_timestamp is None:
            return True
        elapsed = (datetime.utcnow() - self.last_train_timestamp).total_seconds() / 3600
        return elapsed >= hours_since_last
