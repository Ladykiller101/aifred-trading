"""Transformer encoder model for multi-timeframe price prediction.

Architecture:
    Input (batch, seq_len, features) ->
    Positional Encoding ->
    Transformer Encoder (multi-head self-attention) ->
    Temporal aggregation (learnable query) ->
    FC classifier ->
    Output (direction + confidence)

Handles multiple timeframes simultaneously through feature concatenation
and explicit timeframe embedding.
"""

import os
import math
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.types import Signal, Direction

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence ordering."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for time-series classification."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Learnable aggregation query (instead of just taking mean/last)
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.agg_attention = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Magnitude head
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, num_classes)
            confidence: (batch, 1)
            magnitude: (batch, 1)
        """
        # Project to d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        encoded = self.layer_norm(encoded)

        # Learnable aggregation via cross-attention
        batch_size = encoded.size(0)
        query = self.agg_query.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        aggregated, _ = self.agg_attention(query, encoded, encoded)
        aggregated = aggregated.squeeze(1)  # (batch, d_model)

        logits = self.classifier(aggregated)
        confidence = self.confidence_head(aggregated)
        magnitude = self.magnitude_head(aggregated)

        return logits, confidence, magnitude


class TransformerModel:
    """Train/predict interface wrapping the Transformer encoder."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.0005,
        epochs: int = 100,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
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
        """Train the Transformer model.

        Args:
            X: Training sequences (n_samples, seq_len, features).
            y: Training labels (n_samples,).
            X_val: Optional validation sequences.
            y_val: Optional validation labels.
            epochs: Override default epoch count.

        Returns:
            Dict with training history.
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

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits, _, _ = self.model(X_v)
                    val_loss = self.criterion(v_logits, y_v).item()
                self.val_losses.append(val_loss)
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 15:
                    logger.info(f"Transformer early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                val_str = f", val_loss={val_loss:.4f}" if has_val else ""
                logger.info(
                    f"Transformer epoch {epoch + 1}/{epochs}: "
                    f"loss={avg_loss:.4f}{val_str}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_trained = True

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

    def predict(self, X: np.ndarray, asset: str = "", timeframe: str = "1h") -> Signal:
        """Predict from a single sequence.

        Args:
            X: (1, seq_len, features) or (seq_len, features).
            asset: Asset identifier.
            timeframe: Timeframe string.

        Returns:
            Signal with prediction.
        """
        if not self.is_trained:
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=0.0,
                source="transformer",
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

        prob_up = probs[1] if len(probs) > 1 else 0.5
        combined_confidence = float(conf_val * 100)

        if prob_up > 0.65:
            direction = (
                Direction.STRONG_BUY if combined_confidence > 75 else Direction.BUY
            )
        elif prob_up > 0.55:
            direction = Direction.BUY
        elif prob_up < 0.35:
            direction = (
                Direction.STRONG_SELL if combined_confidence > 75 else Direction.SELL
            )
        elif prob_up < 0.45:
            direction = Direction.SELL
        else:
            direction = Direction.HOLD

        return Signal(
            asset=asset,
            direction=direction,
            confidence=combined_confidence,
            source="transformer",
            timeframe=timeframe,
            metadata={
                "prob_up": float(prob_up),
                "prob_down": float(1 - prob_up),
                "raw_confidence": float(conf_val),
                "magnitude": float(mag_val),
            },
        )

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction.

        Args:
            X: (n_samples, seq_len, features)

        Returns:
            (probabilities [n, 2], confidence [n])
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
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "is_trained": self.is_trained,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "timestamp": datetime.utcnow().isoformat(),
            },
            path,
        )
        logger.info(f"Transformer model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model = TransformerEncoder(
            input_size=checkpoint["input_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.is_trained = checkpoint.get("is_trained", True)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Transformer model loaded from {path}")


def build_multi_timeframe_features(
    dfs: Dict[str, np.ndarray],
    lookback: int = 60,
) -> np.ndarray:
    """Concatenate features from multiple timeframes for transformer input.

    Each timeframe's features are resampled/aligned to the primary timeframe's
    index, then concatenated along the feature dimension.

    Args:
        dfs: Dict of timeframe -> feature array (n_samples, n_features).
             Keys like "1h", "4h", "1d".
        lookback: Sequence length for the primary timeframe.

    Returns:
        Combined feature array (n_samples - lookback, lookback, total_features).
        Features from slower timeframes are forward-filled to match the
        primary timeframe's resolution.
    """
    timeframes = sorted(dfs.keys())
    if not timeframes:
        return np.array([])

    primary = dfs[timeframes[0]]
    n_samples = primary.shape[0]

    # Forward-fill slower timeframes to primary resolution
    aligned = []
    for tf in timeframes:
        arr = dfs[tf]
        if arr.shape[0] < n_samples:
            # Repeat each row to match primary resolution
            ratio = n_samples // arr.shape[0]
            arr = np.repeat(arr, ratio, axis=0)[:n_samples]
        elif arr.shape[0] > n_samples:
            arr = arr[:n_samples]
        aligned.append(arr)

    combined = np.concatenate(aligned, axis=1)  # (n_samples, total_features)

    # Build sequences
    seq_count = n_samples - lookback
    if seq_count <= 0:
        return np.array([])

    sequences = np.zeros(
        (seq_count, lookback, combined.shape[1]), dtype=np.float32
    )
    for i in range(seq_count):
        sequences[i] = combined[i : i + lookback]

    return sequences
