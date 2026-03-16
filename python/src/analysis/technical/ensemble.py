"""Ensemble meta-learner combining all model outputs.

Uses XGBoost as a meta-learner (stacking ensemble) to combine predictions
from LSTM, Transformer, Pattern CNN, and indicator rule signals into a
unified trading signal. Auto-weights models based on recent performance.

Features:
- Dynamic weight adjustment using exponential moving average of accuracy
- Signal quality tiering (A+, A, B, C) with minimum 80% ensemble agreement
- Conflict resolution: reduces confidence or defaults to HOLD when models disagree
- Drawdown-aware confidence scaling
- Walk-forward validated weight updates
- Human-readable reasoning generation for every signal
"""

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.utils.types import Signal, Direction

logger = logging.getLogger(__name__)

# Signal quality tiers
SIGNAL_TIER_APLUS = "A+"  # 5+ confluences, >85% confidence
SIGNAL_TIER_A = "A"       # 3-4 confluences, >75% confidence
SIGNAL_TIER_B = "B"       # 2 confluences, >65% confidence
SIGNAL_TIER_C = "C"       # <2 confluences or <65% confidence -- skip

# Minimum ensemble agreement to enter a trade (70%)
MIN_ENSEMBLE_AGREEMENT = 0.80

# EMA decay factor for dynamic weight adjustment (higher = more responsive)
EMA_ALPHA = 0.1

# Minimum trades before EMA weights become active
MIN_TRADES_FOR_EMA = 15


class EnsembleMetaLearner:
    """XGBoost meta-learner combining signals from all sub-models.

    Optimized with:
    - EMA-based dynamic weight adjustment (more responsive than rolling average)
    - Drawdown-aware confidence scaling
    - Walk-forward validated weight updates
    - Strong disagreement detection with confidence penalty
    - Minimum 80% ensemble agreement for trade entry
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        performance_window: int = 100,
        rolling_accuracy_window: int = 30,
        ema_alpha: float = EMA_ALPHA,
        min_ensemble_agreement: float = MIN_ENSEMBLE_AGREEMENT,
    ):
        """
        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            performance_window: Number of recent predictions to track
                for adaptive model weighting.
            rolling_accuracy_window: Window for rolling accuracy tracking
                used in dynamic weight adjustment.
            ema_alpha: Exponential moving average decay factor for weight updates.
                Higher values give more weight to recent performance.
            min_ensemble_agreement: Minimum fraction of models that must agree
                on direction for a trade signal (default 0.70).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.performance_window = performance_window
        self.rolling_accuracy_window = rolling_accuracy_window
        self.ema_alpha = ema_alpha
        self.min_ensemble_agreement = min_ensemble_agreement

        self.meta_model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Track recent performance per model for adaptive weighting
        self.model_performance: Dict[str, deque] = {
            "lstm": deque(maxlen=performance_window),
            "transformer": deque(maxlen=performance_window),
            "pattern_cnn": deque(maxlen=performance_window),
            "rules": deque(maxlen=performance_window),
        }

        # Rolling accuracy tracking for dynamic weight adjustment
        self.rolling_accuracy: Dict[str, deque] = {
            "lstm": deque(maxlen=rolling_accuracy_window),
            "transformer": deque(maxlen=rolling_accuracy_window),
            "pattern_cnn": deque(maxlen=rolling_accuracy_window),
            "rules": deque(maxlen=rolling_accuracy_window),
        }

        # EMA accuracy tracking (exponential moving average)
        self._ema_accuracy: Dict[str, float] = {
            "lstm": 0.5,
            "transformer": 0.5,
            "pattern_cnn": 0.5,
            "rules": 0.5,
        }
        self._ema_trade_count: Dict[str, int] = {
            "lstm": 0,
            "transformer": 0,
            "pattern_cnn": 0,
            "rules": 0,
        }

        # Adaptive weights (start with reasonable priors)
        self.model_weights: Dict[str, float] = {
            "lstm": 0.25,
            "transformer": 0.25,
            "pattern_cnn": 0.20,
            "rules": 0.30,
        }

        # Drawdown state for confidence scaling
        self._current_drawdown: float = 0.0
        self._drawdown_threshold_mild: float = 0.03   # 3% drawdown
        self._drawdown_threshold_severe: float = 0.07  # 7% drawdown

        # Walk-forward validation state
        self._wf_fold_results: List[Dict] = []
        self._wf_validated_weights: Optional[Dict[str, float]] = None

    def set_drawdown(self, drawdown_fraction: float) -> None:
        """Update current drawdown level for confidence scaling.

        Args:
            drawdown_fraction: Current drawdown as a positive fraction (e.g., 0.05 = 5%).
        """
        self._current_drawdown = max(0.0, drawdown_fraction)

    def _drawdown_confidence_scale(self) -> float:
        """Calculate confidence scaling factor based on current drawdown.

        Returns a multiplier between 0.5 and 1.0:
        - No drawdown: 1.0 (full confidence)
        - 3% drawdown: 0.85 (moderate reduction)
        - 7%+ drawdown: 0.60 (aggressive reduction)
        - 10%+ drawdown: 0.50 (minimum confidence)
        """
        dd = self._current_drawdown
        if dd <= 0.01:
            return 1.0
        elif dd <= self._drawdown_threshold_mild:
            # Linear scale from 1.0 to 0.85
            t = dd / self._drawdown_threshold_mild
            return 1.0 - 0.15 * t
        elif dd <= self._drawdown_threshold_severe:
            # Linear scale from 0.85 to 0.60
            t = (dd - self._drawdown_threshold_mild) / (
                self._drawdown_threshold_severe - self._drawdown_threshold_mild
            )
            return 0.85 - 0.25 * t
        else:
            # Beyond severe: scale from 0.60 down to 0.50
            excess = dd - self._drawdown_threshold_severe
            return max(0.50, 0.60 - excess * 2.0)

    def build_meta_features(
        self,
        lstm_signal: Optional[Signal] = None,
        transformer_signal: Optional[Signal] = None,
        cnn_signal: Optional[Signal] = None,
        rule_signal: float = 0.0,
        indicator_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Build feature vector for the meta-learner from sub-model outputs.

        Args:
            lstm_signal: Signal from LSTM model.
            transformer_signal: Signal from Transformer model.
            cnn_signal: Signal from Pattern CNN.
            rule_signal: Composite rule signal value (-1 to 1).
            indicator_features: Optional dict of additional indicator values.

        Returns:
            Feature vector (1D array).
        """
        features = []

        # LSTM features
        if lstm_signal is not None:
            features.extend([
                _direction_to_numeric(lstm_signal.direction),
                lstm_signal.confidence / 100.0,
                lstm_signal.metadata.get("prob_up", 0.5),
                lstm_signal.metadata.get("magnitude", 0.0),
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.0])

        # Transformer features
        if transformer_signal is not None:
            features.extend([
                _direction_to_numeric(transformer_signal.direction),
                transformer_signal.confidence / 100.0,
                transformer_signal.metadata.get("prob_up", 0.5),
                transformer_signal.metadata.get("magnitude", 0.0),
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.0])

        # CNN features
        if cnn_signal is not None:
            features.extend([
                _direction_to_numeric(cnn_signal.direction),
                cnn_signal.confidence / 100.0,
                cnn_signal.metadata.get("prob_up", 0.5),
                cnn_signal.metadata.get("pattern_direction", 0.0),
                len(cnn_signal.metadata.get("detected_patterns", [])) / 5.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.0, 0.0])

        # Rule signal
        features.append(rule_signal)

        # Agreement features (cross-model)
        directions = []
        confidences = []
        if lstm_signal:
            directions.append(_direction_to_numeric(lstm_signal.direction))
            confidences.append(lstm_signal.confidence / 100.0)
        if transformer_signal:
            directions.append(_direction_to_numeric(transformer_signal.direction))
            confidences.append(transformer_signal.confidence / 100.0)
        if cnn_signal:
            directions.append(_direction_to_numeric(cnn_signal.direction))
            confidences.append(cnn_signal.confidence / 100.0)
        directions.append(np.sign(rule_signal))
        confidences.append(abs(rule_signal))

        if directions:
            features.append(np.mean(directions))  # Average direction
            features.append(np.std(directions))  # Disagreement
            # Count how many agree on direction
            bullish = sum(1 for d in directions if d > 0)
            bearish = sum(1 for d in directions if d < 0)
            features.append(bullish / len(directions))
            features.append(bearish / len(directions))
            # Confidence-weighted agreement
            if confidences:
                features.append(float(np.mean(confidences)))
                features.append(float(np.std(confidences)))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Adaptive model weights as features
        features.extend([
            self.model_weights["lstm"],
            self.model_weights["transformer"],
            self.model_weights["pattern_cnn"],
            self.model_weights["rules"],
        ])

        # EMA accuracy as features (gives meta-learner awareness of model quality)
        features.extend([
            self._ema_accuracy["lstm"],
            self._ema_accuracy["transformer"],
            self._ema_accuracy["pattern_cnn"],
            self._ema_accuracy["rules"],
        ])

        # Drawdown state as feature
        features.append(self._current_drawdown)
        features.append(self._drawdown_confidence_scale())

        # Additional indicator features
        if indicator_features:
            for key in sorted(indicator_features.keys()):
                features.append(indicator_features[key])

        return np.nan_to_num(np.array(features, dtype=np.float32), 0.0)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train the meta-learner.

        Args:
            X: Meta-feature matrix (n_samples, n_meta_features).
            y: Labels (n_samples,) -- 0=down, 1=up.
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation labels.

        Returns:
            Training metrics.
        """
        if len(X) < 50:
            logger.warning(f"Insufficient meta-training data: {len(X)} samples")
            return {"status": "skipped", "reason": "insufficient_data"}

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning("Only one class in meta-training data")
            return {"status": "skipped", "reason": "single_class"}

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.meta_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            verbosity=0,
        )

        fit_params = {}
        if X_val is not None and y_val is not None and len(X_val) >= 10:
            X_val_scaled = self.scaler.transform(X_val)
            fit_params["eval_set"] = [(X_val_scaled, y_val)]
            self.meta_model.set_params(early_stopping_rounds=20)

        self.meta_model.fit(X_scaled, y, **fit_params)
        self.is_trained = True

        # Training accuracy
        train_preds = self.meta_model.predict(X_scaled)
        train_accuracy = np.mean(train_preds == y)

        result = {
            "accuracy": float(train_accuracy),
            "n_samples": len(X),
        }

        # Validation accuracy
        if X_val is not None and y_val is not None and len(X_val) >= 10:
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.meta_model.predict(X_val_scaled)
            val_accuracy = np.mean(val_preds == y_val)
            result["val_accuracy"] = float(val_accuracy)
            logger.info(
                "Ensemble meta-learner trained: train_acc=%.3f, val_acc=%.3f, samples=%d",
                train_accuracy, val_accuracy, len(X),
            )
        else:
            logger.info(
                "Ensemble meta-learner trained: accuracy=%.3f, samples=%d",
                train_accuracy, len(X),
            )

        return result

    def walk_forward_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        min_train_size: int = 200,
    ) -> Dict[str, float]:
        """Run walk-forward validation and update weights based on OOS performance.

        Splits data chronologically into expanding train/fixed test windows.
        Updates model weights based on out-of-sample accuracy per model.

        Args:
            X: Full meta-feature matrix (chronologically ordered).
            y: Full label array.
            n_folds: Number of walk-forward folds.
            min_train_size: Minimum training window size.

        Returns:
            Validation metrics including per-fold accuracy.
        """
        n = len(X)
        if n < min_train_size + 50:
            logger.warning("Insufficient data for walk-forward validation: %d samples", n)
            return {"status": "skipped", "reason": "insufficient_data"}

        test_size = (n - min_train_size) // n_folds
        if test_size < 20:
            test_size = 20
            n_folds = max(1, (n - min_train_size) // test_size)

        fold_results = []
        for fold in range(n_folds):
            train_end = min_train_size + fold * test_size
            test_end = min(train_end + test_size, n)

            if train_end >= n or test_end <= train_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_train)) < 2 or len(X_test) < 5:
                continue

            # Train a temporary model on this fold
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X_train_s, y_train)

            test_preds = model.predict(X_test_s)
            fold_acc = float(np.mean(test_preds == y_test))

            fold_results.append({
                "fold": fold,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": fold_acc,
            })

        if not fold_results:
            return {"status": "skipped", "reason": "no_valid_folds"}

        self._wf_fold_results = fold_results
        accuracies = [r["accuracy"] for r in fold_results]
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))

        # Use the latest fold accuracy to adjust weights
        latest_acc = fold_results[-1]["accuracy"]
        if latest_acc < 0.52:
            logger.warning(
                "Walk-forward OOS accuracy poor (%.3f). Consider retraining.",
                latest_acc,
            )

        logger.info(
            "Walk-forward validation: mean_acc=%.3f +/- %.3f over %d folds",
            mean_acc, std_acc, len(fold_results),
        )

        return {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "n_folds": len(fold_results),
            "fold_details": fold_results,
            "latest_fold_accuracy": latest_acc,
        }

    def predict(
        self,
        lstm_signal: Optional[Signal] = None,
        transformer_signal: Optional[Signal] = None,
        cnn_signal: Optional[Signal] = None,
        rule_signal: float = 0.0,
        indicator_features: Optional[Dict[str, float]] = None,
        asset: str = "",
        timeframe: str = "1h",
    ) -> Signal:
        """Generate unified signal from all sub-model predictions.

        If the meta-learner is not trained, falls back to weighted averaging.
        Applies conflict resolution, signal quality tiering, drawdown scaling,
        minimum ensemble agreement check, and reasoning.

        Args:
            lstm_signal: LSTM prediction.
            transformer_signal: Transformer prediction.
            cnn_signal: Pattern CNN prediction.
            rule_signal: Rule-based signal (-1 to 1).
            indicator_features: Additional indicator values.
            asset: Asset identifier.
            timeframe: Timeframe string.

        Returns:
            Unified Signal with direction, confidence, quality tier, and reasoning.
        """
        meta_features = self.build_meta_features(
            lstm_signal, transformer_signal, cnn_signal,
            rule_signal, indicator_features,
        )

        # Check for model conflicts before generating signal
        conflict_result = self._check_conflicts(
            lstm_signal, transformer_signal, cnn_signal, rule_signal
        )

        # Check ensemble agreement
        agreement_result = self._check_ensemble_agreement(
            lstm_signal, transformer_signal, cnn_signal, rule_signal
        )

        # Strong disagreement: reduce confidence or skip entirely
        if conflict_result["has_conflict"]:
            if conflict_result.get("strong_conflict", False) or not self.is_trained:
                reasoning = generate_reasoning(
                    lstm_signal, transformer_signal, cnn_signal,
                    rule_signal, indicator_features,
                    direction=Direction.HOLD,
                    method="conflict_resolution",
                    conflict_info=conflict_result,
                )
                return Signal(
                    asset=asset,
                    direction=Direction.HOLD,
                    confidence=0.0,
                    source="ensemble",
                    timeframe=timeframe,
                    metadata={
                        "method": "conflict_resolution",
                        "conflict_info": conflict_result,
                        "signal_tier": SIGNAL_TIER_C,
                        "reasoning": reasoning,
                        "model_weights": dict(self.model_weights),
                        "ema_accuracy": dict(self._ema_accuracy),
                        "drawdown": self._current_drawdown,
                        "drawdown_scale": self._drawdown_confidence_scale(),
                        "sub_signals": _collect_sub_signals(
                            lstm_signal, transformer_signal, cnn_signal, rule_signal
                        ),
                    },
                )

        # Check minimum ensemble agreement (70% must agree)
        if not agreement_result["meets_threshold"]:
            reasoning = generate_reasoning(
                lstm_signal, transformer_signal, cnn_signal,
                rule_signal, indicator_features,
                direction=Direction.HOLD,
                method="insufficient_agreement",
                conflict_info=agreement_result,
            )
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=min(agreement_result["agreement_pct"] * 100, 50.0),
                source="ensemble",
                timeframe=timeframe,
                metadata={
                    "method": "insufficient_agreement",
                    "agreement_info": agreement_result,
                    "signal_tier": SIGNAL_TIER_C,
                    "reasoning": reasoning,
                    "model_weights": dict(self.model_weights),
                    "sub_signals": _collect_sub_signals(
                        lstm_signal, transformer_signal, cnn_signal, rule_signal
                    ),
                },
            )

        if self.is_trained and self.meta_model is not None:
            signal = self._predict_with_model(
                meta_features, lstm_signal, transformer_signal,
                cnn_signal, rule_signal, asset, timeframe,
            )
        else:
            signal = self._predict_fallback(
                lstm_signal, transformer_signal, cnn_signal,
                rule_signal, asset, timeframe,
            )

        # Apply drawdown-aware confidence scaling
        dd_scale = self._drawdown_confidence_scale()
        if dd_scale < 1.0:
            original_confidence = signal.confidence
            signal.confidence *= dd_scale
            signal.metadata["drawdown_scale"] = dd_scale
            signal.metadata["original_confidence"] = original_confidence
            signal.metadata["drawdown"] = self._current_drawdown

        # Apply signal quality tiering
        confluences = self._count_confluences(
            lstm_signal, transformer_signal, cnn_signal, rule_signal
        )
        tier = self._determine_signal_tier(confluences, signal.confidence)
        signal.metadata["signal_tier"] = tier
        signal.metadata["confluences"] = confluences
        signal.metadata["ensemble_agreement"] = agreement_result

        # Generate reasoning
        reasoning = generate_reasoning(
            lstm_signal, transformer_signal, cnn_signal,
            rule_signal, indicator_features,
            direction=signal.direction,
            method=signal.metadata.get("method", "unknown"),
            tier=tier,
            confluences=confluences,
            drawdown=self._current_drawdown,
            drawdown_scale=dd_scale,
        )
        signal.metadata["reasoning"] = reasoning

        # Downgrade to HOLD for C-tier signals (skip -- don't trade)
        if tier == SIGNAL_TIER_C:
            signal.direction = Direction.HOLD
            signal.confidence = min(signal.confidence, 50.0)

        return signal

    def _predict_with_model(
        self,
        meta_features: np.ndarray,
        lstm_signal: Optional[Signal],
        transformer_signal: Optional[Signal],
        cnn_signal: Optional[Signal],
        rule_signal: float,
        asset: str,
        timeframe: str,
    ) -> Signal:
        """Predict using trained meta-model."""
        X = self.scaler.transform(meta_features.reshape(1, -1))
        prob = self.meta_model.predict_proba(X)[0]

        prob_up = prob[1] if len(prob) > 1 else 0.5

        # Confidence from probability distance from 0.5
        confidence = abs(prob_up - 0.5) * 200  # Scale to 0-100

        direction = _prob_to_direction(prob_up, confidence)

        return Signal(
            asset=asset,
            direction=direction,
            confidence=float(confidence),
            source="ensemble",
            timeframe=timeframe,
            metadata={
                "prob_up": float(prob_up),
                "method": "xgboost_meta",
                "model_weights": dict(self.model_weights),
                "ema_accuracy": dict(self._ema_accuracy),
                "sub_signals": _collect_sub_signals(
                    lstm_signal, transformer_signal, cnn_signal, rule_signal
                ),
            },
        )

    def _predict_fallback(
        self,
        lstm_signal: Optional[Signal],
        transformer_signal: Optional[Signal],
        cnn_signal: Optional[Signal],
        rule_signal: float,
        asset: str,
        timeframe: str,
    ) -> Signal:
        """Fallback: weighted average of sub-model directions."""
        weighted_sum = 0.0
        weight_total = 0.0
        confidences = []

        if lstm_signal is not None:
            w = self.model_weights["lstm"]
            weighted_sum += _direction_to_numeric(lstm_signal.direction) * w
            weight_total += w
            confidences.append(lstm_signal.confidence)

        if transformer_signal is not None:
            w = self.model_weights["transformer"]
            weighted_sum += _direction_to_numeric(transformer_signal.direction) * w
            weight_total += w
            confidences.append(transformer_signal.confidence)

        if cnn_signal is not None:
            w = self.model_weights["pattern_cnn"]
            weighted_sum += _direction_to_numeric(cnn_signal.direction) * w
            weight_total += w
            confidences.append(cnn_signal.confidence)

        w = self.model_weights["rules"]
        weighted_sum += rule_signal * w
        weight_total += w
        confidences.append(abs(rule_signal) * 100)

        if weight_total > 0:
            avg_direction = weighted_sum / weight_total
        else:
            avg_direction = 0.0

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        direction = _numeric_to_direction(avg_direction, avg_confidence)

        return Signal(
            asset=asset,
            direction=direction,
            confidence=avg_confidence,
            source="ensemble",
            timeframe=timeframe,
            metadata={
                "avg_direction": float(avg_direction),
                "method": "weighted_average",
                "model_weights": dict(self.model_weights),
                "ema_accuracy": dict(self._ema_accuracy),
                "sub_signals": _collect_sub_signals(
                    lstm_signal, transformer_signal, cnn_signal, rule_signal
                ),
            },
        )

    def update_performance(
        self,
        model_name: str,
        was_correct: bool,
    ) -> None:
        """Record a prediction outcome for adaptive weighting.

        Uses exponential moving average (EMA) for fast adaptation to regime
        changes, plus rolling window for smoothing.

        Args:
            model_name: One of "lstm", "transformer", "pattern_cnn", "rules".
            was_correct: Whether the model's prediction was correct.
        """
        outcome = 1.0 if was_correct else 0.0

        if model_name in self.model_performance:
            self.model_performance[model_name].append(outcome)
        if model_name in self.rolling_accuracy:
            self.rolling_accuracy[model_name].append(outcome)

        # Update EMA accuracy
        if model_name in self._ema_accuracy:
            self._ema_trade_count[model_name] += 1
            count = self._ema_trade_count[model_name]

            if count <= 1:
                self._ema_accuracy[model_name] = outcome
            else:
                # EMA: new_value = alpha * observation + (1 - alpha) * old_value
                # Adaptive alpha: faster adaptation when accuracy is poor
                recent_acc = self._ema_accuracy[model_name]
                alpha = 0.3 if recent_acc < 0.55 else 0.1
                self._ema_accuracy[model_name] = (
                    alpha * outcome + (1 - alpha) * self._ema_accuracy[model_name]
                )

        self._update_weights()

    def _update_weights(self) -> None:
        """Recalculate model weights based on EMA accuracy.

        Uses EMA accuracy for responsive adaptation when enough data exists;
        falls back to rolling average or priors otherwise.
        """
        accuracies = {}
        for name in self.model_weights.keys():
            count = self._ema_trade_count.get(name, 0)

            if count >= MIN_TRADES_FOR_EMA:
                # Primary: use EMA accuracy (most responsive)
                accuracies[name] = self._ema_accuracy[name]
            else:
                # Fallback: use rolling average if available
                rolling = self.rolling_accuracy.get(name, deque())
                long_term = self.model_performance.get(name, deque())

                if len(rolling) >= 10:
                    accuracies[name] = np.mean(list(rolling))
                elif len(long_term) >= 10:
                    accuracies[name] = np.mean(list(long_term))
                else:
                    accuracies[name] = 0.5  # Prior: 50% accuracy

        # Softmax-like weighting (temperature=2 for smoothing)
        temperature = 2.0
        values = np.array([accuracies[k] for k in self.model_weights.keys()])
        exp_vals = np.exp(values / temperature)
        weights = exp_vals / exp_vals.sum()

        for i, name in enumerate(self.model_weights.keys()):
            self.model_weights[name] = float(weights[i])

    def apply_walk_forward_weights(
        self,
        fold_model_accuracies: Dict[str, List[float]],
    ) -> None:
        """Apply weight adjustments validated through walk-forward results.

        Uses out-of-sample accuracy from walk-forward folds to set model
        weights. This prevents overfitting to in-sample performance.

        Args:
            fold_model_accuracies: Dict mapping model_name -> list of
                per-fold OOS accuracies.
        """
        validated_accuracies = {}
        for name, accs in fold_model_accuracies.items():
            if name in self.model_weights and len(accs) > 0:
                # Use mean of last 3 folds (most recent out-of-sample)
                recent_accs = accs[-3:]
                validated_accuracies[name] = float(np.mean(recent_accs))

        if not validated_accuracies:
            return

        # Only update if we have data for all models
        if len(validated_accuracies) < len(self.model_weights):
            # Fill missing with current EMA
            for name in self.model_weights:
                if name not in validated_accuracies:
                    validated_accuracies[name] = self._ema_accuracy.get(name, 0.5)

        # Softmax with temperature
        temperature = 2.0
        values = np.array([validated_accuracies[k] for k in self.model_weights.keys()])
        exp_vals = np.exp(values / temperature)
        weights = exp_vals / exp_vals.sum()

        self._wf_validated_weights = {}
        for i, name in enumerate(self.model_weights.keys()):
            self._wf_validated_weights[name] = float(weights[i])
            self.model_weights[name] = float(weights[i])

        logger.info(
            "Walk-forward validated weights applied: %s (from accuracies: %s)",
            self.model_weights, validated_accuracies,
        )

    def _check_conflicts(
        self,
        lstm_signal: Optional[Signal],
        transformer_signal: Optional[Signal],
        cnn_signal: Optional[Signal],
        rule_signal: float,
    ) -> Dict:
        """Check if sub-models are in significant conflict.

        Detects both standard conflicts (roughly equal split) and
        strong conflicts (high-confidence models disagreeing).

        Returns conflict info dict with details.
        """
        directions = []
        model_names = []
        model_confidences = []

        if lstm_signal and lstm_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(lstm_signal.direction))
            model_names.append("lstm")
            model_confidences.append(lstm_signal.confidence)
        if transformer_signal and transformer_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(transformer_signal.direction))
            model_names.append("transformer")
            model_confidences.append(transformer_signal.confidence)
        if cnn_signal and cnn_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(cnn_signal.direction))
            model_names.append("pattern_cnn")
            model_confidences.append(cnn_signal.confidence)
        if abs(rule_signal) > 0.1:
            directions.append(np.sign(rule_signal))
            model_names.append("rules")
            model_confidences.append(abs(rule_signal) * 100)

        if len(directions) < 2:
            return {"has_conflict": False, "strong_conflict": False, "n_models": len(directions)}

        bullish = sum(1 for d in directions if d > 0)
        bearish = sum(1 for d in directions if d < 0)
        total = len(directions)

        # Standard conflict if roughly equal split
        has_conflict = (
            bullish > 0 and bearish > 0
            and min(bullish, bearish) / total >= 0.4
        )

        # Strong conflict: high-confidence models on opposite sides
        strong_conflict = False
        if bullish > 0 and bearish > 0:
            bullish_confs = [
                model_confidences[i] for i, d in enumerate(directions) if d > 0
            ]
            bearish_confs = [
                model_confidences[i] for i, d in enumerate(directions) if d < 0
            ]
            max_bull = max(bullish_confs) if bullish_confs else 0
            max_bear = max(bearish_confs) if bearish_confs else 0

            # Both sides have high-confidence models
            if max_bull > 70 and max_bear > 70:
                strong_conflict = True
                logger.warning(
                    "Strong model conflict: bullish (max conf %.1f%%) vs "
                    "bearish (max conf %.1f%%)",
                    max_bull, max_bear,
                )

        return {
            "has_conflict": has_conflict,
            "strong_conflict": strong_conflict,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "n_models": total,
            "models_bullish": [
                model_names[i] for i, d in enumerate(directions) if d > 0
            ],
            "models_bearish": [
                model_names[i] for i, d in enumerate(directions) if d < 0
            ],
        }

    def _check_ensemble_agreement(
        self,
        lstm_signal: Optional[Signal],
        transformer_signal: Optional[Signal],
        cnn_signal: Optional[Signal],
        rule_signal: float,
    ) -> Dict:
        """Check if minimum 70% of models agree on the majority direction.

        Returns agreement info dict.
        """
        directions = []
        if lstm_signal and lstm_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(lstm_signal.direction))
        if transformer_signal and transformer_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(transformer_signal.direction))
        if cnn_signal and cnn_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(cnn_signal.direction))
        if abs(rule_signal) > 0.1:
            directions.append(np.sign(rule_signal))

        total = len(directions)
        if total == 0:
            return {
                "meets_threshold": False,
                "agreement_pct": 0.0,
                "majority_direction": "neutral",
                "n_models": 0,
            }

        bullish = sum(1 for d in directions if d > 0)
        bearish = sum(1 for d in directions if d < 0)
        majority = max(bullish, bearish)
        agreement_pct = majority / total

        majority_dir = "bullish" if bullish >= bearish else "bearish"

        meets = agreement_pct >= self.min_ensemble_agreement

        return {
            "meets_threshold": meets,
            "agreement_pct": agreement_pct,
            "majority_direction": majority_dir,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "n_models": total,
            "threshold": self.min_ensemble_agreement,
        }

    def _count_confluences(
        self,
        lstm_signal: Optional[Signal],
        transformer_signal: Optional[Signal],
        cnn_signal: Optional[Signal],
        rule_signal: float,
    ) -> int:
        """Count how many sub-models agree on the majority direction."""
        directions = []
        if lstm_signal and lstm_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(lstm_signal.direction))
        if transformer_signal and transformer_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(transformer_signal.direction))
        if cnn_signal and cnn_signal.direction != Direction.HOLD:
            directions.append(_direction_to_numeric(cnn_signal.direction))
        if abs(rule_signal) > 0.1:
            directions.append(np.sign(rule_signal))

        if not directions:
            return 0

        bullish = sum(1 for d in directions if d > 0)
        bearish = sum(1 for d in directions if d < 0)
        return max(bullish, bearish)

    def _determine_signal_tier(self, confluences: int, confidence: float) -> str:
        """Determine signal quality tier based on confluences and confidence.

        A+: 5+ confluences, >85% confidence (forward-compatible for more models)
        A:  3-4 confluences, >75% confidence
        B:  2 confluences, >65% confidence
        C:  <2 confluences or <65% confidence -- skip (don't trade)
        """
        if confluences >= 4 and confidence > 85:
            return SIGNAL_TIER_APLUS
        elif confluences >= 3 and confidence > 75:
            return SIGNAL_TIER_A
        elif confluences >= 2 and confidence > 65:
            return SIGNAL_TIER_B
        else:
            return SIGNAL_TIER_C

    def get_model_weights(self) -> Dict[str, float]:
        """Return current adaptive model weights."""
        return dict(self.model_weights)

    def get_model_accuracy(self) -> Dict[str, float]:
        """Return recent accuracy for each model."""
        result = {}
        for name, history in self.model_performance.items():
            if len(history) > 0:
                result[name] = float(np.mean(list(history)))
            else:
                result[name] = 0.0
        return result

    def get_rolling_accuracy(self) -> Dict[str, float]:
        """Return rolling 30-trade accuracy for each model."""
        result = {}
        for name, history in self.rolling_accuracy.items():
            if len(history) > 0:
                result[name] = float(np.mean(list(history)))
            else:
                result[name] = 0.0
        return result

    def get_ema_accuracy(self) -> Dict[str, float]:
        """Return EMA accuracy for each model."""
        return dict(self._ema_accuracy)

    def get_diagnostics(self) -> Dict:
        """Return full ensemble diagnostics for monitoring."""
        return {
            "model_weights": dict(self.model_weights),
            "ema_accuracy": dict(self._ema_accuracy),
            "ema_trade_counts": dict(self._ema_trade_count),
            "rolling_accuracy": self.get_rolling_accuracy(),
            "long_term_accuracy": self.get_model_accuracy(),
            "is_trained": self.is_trained,
            "drawdown": self._current_drawdown,
            "drawdown_scale": self._drawdown_confidence_scale(),
            "wf_validated_weights": self._wf_validated_weights,
            "wf_fold_count": len(self._wf_fold_results),
        }


def _direction_to_numeric(direction: Direction) -> float:
    """Convert Direction enum to numeric value."""
    mapping = {
        Direction.STRONG_BUY: 1.0,
        Direction.BUY: 0.5,
        Direction.HOLD: 0.0,
        Direction.SELL: -0.5,
        Direction.STRONG_SELL: -1.0,
    }
    return mapping.get(direction, 0.0)


def _prob_to_direction(prob_up: float, confidence: float) -> Direction:
    """Convert probability and confidence to Direction."""
    if prob_up > 0.65:
        return Direction.STRONG_BUY if confidence > 75 else Direction.BUY
    elif prob_up > 0.55:
        return Direction.BUY
    elif prob_up < 0.35:
        return Direction.STRONG_SELL if confidence > 75 else Direction.SELL
    elif prob_up < 0.45:
        return Direction.SELL
    return Direction.HOLD


def _numeric_to_direction(value: float, confidence: float) -> Direction:
    """Convert numeric direction value to Direction enum."""
    if value > 0.4:
        return Direction.STRONG_BUY if confidence > 75 else Direction.BUY
    elif value > 0.15:
        return Direction.BUY
    elif value < -0.4:
        return Direction.STRONG_SELL if confidence > 75 else Direction.SELL
    elif value < -0.15:
        return Direction.SELL
    return Direction.HOLD


def _collect_sub_signals(
    lstm_signal: Optional[Signal],
    transformer_signal: Optional[Signal],
    cnn_signal: Optional[Signal],
    rule_signal: float,
) -> Dict:
    """Collect sub-model signal summaries for metadata."""
    result = {}
    if lstm_signal:
        result["lstm"] = {
            "direction": lstm_signal.direction.value,
            "confidence": lstm_signal.confidence,
        }
    if transformer_signal:
        result["transformer"] = {
            "direction": transformer_signal.direction.value,
            "confidence": transformer_signal.confidence,
        }
    if cnn_signal:
        result["pattern_cnn"] = {
            "direction": cnn_signal.direction.value,
            "confidence": cnn_signal.confidence,
            "patterns": cnn_signal.metadata.get("detected_patterns", []),
        }
    result["rules"] = {"signal": rule_signal}
    return result


def generate_reasoning(
    lstm_signal: Optional[Signal] = None,
    transformer_signal: Optional[Signal] = None,
    cnn_signal: Optional[Signal] = None,
    rule_signal: float = 0.0,
    indicator_features: Optional[Dict[str, float]] = None,
    direction: Direction = Direction.HOLD,
    method: str = "unknown",
    tier: str = SIGNAL_TIER_C,
    confluences: int = 0,
    conflict_info: Optional[Dict] = None,
    drawdown: float = 0.0,
    drawdown_scale: float = 1.0,
) -> str:
    """Generate a human-readable reasoning string explaining the signal.

    Args:
        lstm_signal: LSTM model signal.
        transformer_signal: Transformer model signal.
        cnn_signal: Pattern CNN signal.
        rule_signal: Rule-based signal value.
        indicator_features: Key indicator values.
        direction: Final signal direction.
        method: Ensemble method used.
        tier: Signal quality tier.
        confluences: Number of confluences.
        conflict_info: Conflict resolution details.
        drawdown: Current drawdown fraction.
        drawdown_scale: Confidence scaling factor from drawdown.

    Returns:
        Human-readable string explaining why this signal was generated.
    """
    parts = []

    parts.append(f"Signal: {direction.value} | Tier: {tier} | Method: {method}")
    parts.append(f"Confluences: {confluences}/4 models agree")

    # Drawdown info
    if drawdown > 0.01:
        parts.append(
            f"Drawdown: {drawdown:.1%} (confidence scaled to {drawdown_scale:.0%})"
        )

    # Conflict info
    if conflict_info and conflict_info.get("has_conflict"):
        bullish_models = ", ".join(conflict_info.get("models_bullish", []))
        bearish_models = ", ".join(conflict_info.get("models_bearish", []))
        strong = " [STRONG]" if conflict_info.get("strong_conflict") else ""
        parts.append(
            f"CONFLICT{strong}: Bullish [{bullish_models}] vs Bearish [{bearish_models}] "
            f"-> Defaulting to HOLD"
        )

    # Insufficient agreement
    if conflict_info and "agreement_pct" in conflict_info:
        pct = conflict_info["agreement_pct"]
        threshold = conflict_info.get("threshold", MIN_ENSEMBLE_AGREEMENT)
        if pct < threshold:
            parts.append(
                f"Agreement: {pct:.0%} (below {threshold:.0%} threshold) -> HOLD"
            )

    # Individual model predictions
    if lstm_signal:
        parts.append(
            f"LSTM: {lstm_signal.direction.value} "
            f"(conf={lstm_signal.confidence:.1f}%, "
            f"prob_up={lstm_signal.metadata.get('prob_up', 0.5):.2f})"
        )

    if transformer_signal:
        parts.append(
            f"Transformer: {transformer_signal.direction.value} "
            f"(conf={transformer_signal.confidence:.1f}%, "
            f"prob_up={transformer_signal.metadata.get('prob_up', 0.5):.2f})"
        )

    if cnn_signal:
        patterns = cnn_signal.metadata.get("detected_patterns", [])
        pattern_names = [p.get("pattern", "?") for p in patterns] if patterns else ["none"]
        parts.append(
            f"PatternCNN: {cnn_signal.direction.value} "
            f"(conf={cnn_signal.confidence:.1f}%, "
            f"patterns={', '.join(pattern_names)})"
        )

    parts.append(f"Rules: signal={rule_signal:.3f}")

    # Key indicators
    if indicator_features:
        indicators_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(indicator_features.items())
            if v is not None
        )
        if indicators_str:
            parts.append(f"Key indicators: {indicators_str}")

    return " | ".join(parts)
