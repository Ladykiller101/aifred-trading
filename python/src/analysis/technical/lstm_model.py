"""Multi-layer LSTM with attention mechanism for price direction prediction.

Architecture:
    Input (batch, lookback, features) ->
    LSTM layers (stacked, with dropout) ->
    Attention mechanism (learns which timesteps matter) ->
    FC layers ->
    Output (direction probability + magnitude + confidence)
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.types import Signal, Direction

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Additive attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — attention-weighted sum
            weights: (batch, seq_len) — attention weights
        """
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class LSTMNetwork(nn.Module):
    """Stacked LSTM with attention for time-series classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Confidence head (separate from class logits)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Magnitude head (expected % move)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, lookback, features)

        Returns:
            logits: (batch, num_classes)
            confidence: (batch, 1) in [0, 1]
            magnitude: (batch, 1) predicted move magnitude
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        context, _ = self.attention(lstm_out)  # (batch, hidden)
        context = self.dropout(context)

        logits = self.classifier(context)
        confidence = self.confidence_head(context)
        magnitude = self.magnitude_head(context)

        return logits, confidence, magnitude


class LSTMModel:
    """Train/predict interface wrapping the LSTM network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.train_losses: list = []
        self.val_losses: list = []
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """Train the LSTM model.

        Args:
            X: Training sequences (n_samples, lookback, features).
            y: Training labels (n_samples,).
            X_val: Optional validation sequences.
            y_val: Optional validation labels.
            epochs: Override default epoch count.

        Returns:
            Dict with 'train_loss' and 'val_loss' history.
        """
        if epochs is None:
            epochs = self.epochs

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.LongTensor(y_val).to(self.device)

        self.train_losses = []
        self.val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits, _, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits, _, _ = self.model(X_v)
                    val_loss = self.criterion(v_logits, y_v).item()
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                val_str = f", val_loss={val_loss:.4f}" if has_val else ""
                logger.info(
                    f"LSTM epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f}{val_str}"
                )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_trained = True

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

    def predict(self, X: np.ndarray, asset: str = "", timeframe: str = "1h") -> Signal:
        """Predict direction, confidence, and magnitude from a single sequence.

        Args:
            X: Input sequence (1, lookback, features) or (lookback, features).
            asset: Asset identifier for the Signal.
            timeframe: Timeframe string.

        Returns:
            Signal with direction, confidence, and metadata.
        """
        if not self.is_trained:
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=0.0,
                source="lstm",
                timeframe=timeframe,
            )

        if X.ndim == 2:
            X = X[np.newaxis, :]

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits, confidence, magnitude = self.model(X_t)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            conf_val = confidence.cpu().numpy()[0, 0]
            mag_val = magnitude.cpu().numpy()[0, 0]

        # Determine direction from probabilities
        prob_up = probs[1] if len(probs) > 1 else 0.5
        combined_confidence = float(conf_val * 100)

        if prob_up > 0.65:
            direction = Direction.STRONG_BUY if combined_confidence > 75 else Direction.BUY
        elif prob_up > 0.55:
            direction = Direction.BUY
        elif prob_up < 0.35:
            direction = Direction.STRONG_SELL if combined_confidence > 75 else Direction.SELL
        elif prob_up < 0.45:
            direction = Direction.SELL
        else:
            direction = Direction.HOLD

        return Signal(
            asset=asset,
            direction=direction,
            confidence=combined_confidence,
            source="lstm",
            timeframe=timeframe,
            metadata={
                "prob_up": float(prob_up),
                "prob_down": float(1 - prob_up),
                "raw_confidence": float(conf_val),
                "magnitude": float(mag_val),
            },
        )

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction returning probabilities and confidence.

        Args:
            X: (n_samples, lookback, features)

        Returns:
            Tuple of (probabilities [n, 2], confidence [n])
        """
        if not self.is_trained:
            n = X.shape[0]
            return np.full((n, 2), 0.5), np.zeros(n)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits, confidence, _ = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            conf = confidence.cpu().numpy().squeeze()

        return probs, conf

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "is_trained": self.is_trained,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "timestamp": datetime.utcnow().isoformat(),
            },
            path,
        )
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model = LSTMNetwork(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.is_trained = checkpoint.get("is_trained", True)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"LSTM model loaded from {path}")
