"""Central orchestrator that coordinates all trading agents.

Runs a configurable scan loop that:
1. Collects signals from Technical Analysis and Sentiment agents
2. Fuses signals using weighted ensemble (default 60% tech, 40% sentiment)
3. Applies confidence threshold (default 70%)
4. Routes approved signals through Risk Management for sizing/stops
5. Routes approved+sized orders through Execution agent
6. Logs all decisions with full reasoning
7. Handles errors gracefully (one agent failing does not crash the system)
8. Supports paper trading mode by default
9. Includes circuit breakers (max daily trades, max daily loss, etc.)
"""

import asyncio
import logging
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.sentiment.sentiment_signals import SentimentAnalysisAgent
from src.analysis.technical.signals import TechnicalAnalysisAgent
from src.config import get_config
from src.execution.execution_engine import ExecutionAgent
from src.monitoring.trade_logger import TradeLogger
from src.monitoring.telegram_alerts import AlertType, TelegramAlerts
from src.risk.correlation_tracker import CorrelationTracker
from src.risk.drawdown_manager import DrawdownManager
from src.risk.portfolio_monitor import PortfolioMonitor
from src.risk.risk_gate import RiskGate
from src.risk.stop_manager import calculate_stop_loss, calculate_take_profit
from src.risk.volatility_regime import detect_regime, get_regime_adjustments
from src.utils.types import (
    AssetClass,
    Direction,
    OrderType,
    PortfolioState,
    Signal,
    TradeProposal,
    TradeResult,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breakers to prevent runaway trading.

    Tracks daily trade counts, daily P&L, consecutive failures,
    and enforces hard limits.
    """

    def __init__(self, config: Dict[str, Any]):
        orch_cfg = config.get("orchestrator", {})
        risk_cfg = config.get("risk", {})
        exec_cfg = config.get("execution", {})

        self.max_daily_trades = orch_cfg.get("max_trades_per_day", 20)
        self.max_daily_loss_pct = risk_cfg.get("max_daily_drawdown_pct", 5.0)
        self.max_consecutive_failures = exec_cfg.get("max_consecutive_failures", 3)

        self._daily_trade_count: int = 0
        self._daily_pnl: float = 0.0
        self._portfolio_value: float = 0.0
        self._consecutive_failures: int = 0
        self._last_reset_date: Optional[datetime] = None
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_until: Optional[datetime] = None

    def reset_daily(self, portfolio_value: float) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self._daily_trade_count = 0
        self._daily_pnl = 0.0
        self._portfolio_value = portfolio_value
        self._last_reset_date = datetime.utcnow()
        # Clear trip if it was daily
        if self._tripped and self._trip_until and datetime.utcnow() >= self._trip_until:
            self._tripped = False
            self._trip_reason = ""
            self._trip_until = None
            logger.info("Circuit breaker reset after cooldown expired")

    def record_trade(self, pnl: float, success: bool) -> None:
        """Record a trade outcome."""
        self._daily_trade_count += 1
        self._daily_pnl += pnl

        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

    def check(self) -> tuple:
        """Check all circuit breakers.

        Returns:
            (tripped: bool, reason: str)
        """
        # Check existing trip with cooldown
        if self._tripped:
            if self._trip_until and datetime.utcnow() >= self._trip_until:
                self._tripped = False
                self._trip_reason = ""
                self._trip_until = None
                logger.info("Circuit breaker cooldown expired, resuming")
            else:
                return True, self._trip_reason

        # Check daily trade limit
        if self._daily_trade_count >= self.max_daily_trades:
            self._trip("max_daily_trades", hours=4)
            return True, (
                f"Daily trade limit reached: {self._daily_trade_count}/{self.max_daily_trades}"
            )

        # Check daily loss limit
        if self._portfolio_value > 0:
            daily_loss_pct = abs(min(0, self._daily_pnl)) / self._portfolio_value * 100
            if daily_loss_pct >= self.max_daily_loss_pct:
                self._trip("max_daily_loss", hours=24)
                return True, (
                    f"Daily loss limit reached: {daily_loss_pct:.2f}% "
                    f"(limit: {self.max_daily_loss_pct:.1f}%)"
                )

        # Check consecutive execution failures
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._trip("consecutive_failures", hours=1)
            return True, (
                f"Consecutive execution failures: {self._consecutive_failures} "
                f"(limit: {self.max_consecutive_failures})"
            )

        return False, "OK"

    def _trip(self, reason_type: str, hours: int = 4) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = f"Circuit breaker tripped: {reason_type}"
        self._trip_until = datetime.utcnow() + timedelta(hours=hours)
        logger.warning(
            "CIRCUIT BREAKER TRIPPED: %s. Cooldown until %s",
            reason_type, self._trip_until.isoformat(),
        )

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "tripped": self._tripped,
            "reason": self._trip_reason,
            "trip_until": self._trip_until.isoformat() if self._trip_until else None,
            "daily_trades": self._daily_trade_count,
            "max_daily_trades": self.max_daily_trades,
            "daily_pnl": self._daily_pnl,
            "consecutive_failures": self._consecutive_failures,
        }


class Orchestrator:
    """Central coordinator for the multi-agent trading system.

    Runs a scan loop at configurable intervals, collects signals from
    all analysis agents, fuses them, applies risk management, and
    routes approved trades through execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the orchestrator with full configuration.

        Args:
            config: Full configuration dict (from default.yaml).
        """
        self.config = config
        orch_cfg = config.get("orchestrator", {})

        # Core parameters
        self.scan_interval = orch_cfg.get("scan_interval_seconds", 60)
        self.min_confidence = orch_cfg.get("min_confidence_threshold", 78)
        self.signal_weights = orch_cfg.get("signal_weights", {
            "technical": 0.60,
            "sentiment": 0.40,
        })
        self.max_daily_trades = orch_cfg.get("max_trades_per_day", 20)

        # Trading mode
        exec_cfg = config.get("execution", {})
        self._paper_mode = exec_cfg.get("mode", "paper") == "paper"

        # Assets to scan
        self._assets = self._build_asset_list(config)

        # Initialize agents (with graceful error handling)
        self._tech_agent: Optional[TechnicalAnalysisAgent] = None
        self._sentiment_agent: Optional[SentimentAnalysisAgent] = None
        self._risk_gate: Optional[RiskGate] = None
        self._execution_agent: Optional[ExecutionAgent] = None
        self._trade_logger: Optional[TradeLogger] = None
        self._telegram: Optional[TelegramAlerts] = None
        self._drawdown_manager: Optional[DrawdownManager] = None
        self._portfolio_monitor: Optional[PortfolioMonitor] = None
        self._correlation_tracker: Optional[CorrelationTracker] = None

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(config)

        # State
        self._running = False
        self._scan_count = 0
        self._last_scan_time: Optional[datetime] = None
        self._daily_trade_count = 0
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._decision_log: List[Dict[str, Any]] = []

        # Data provider callback (set externally)
        self._data_provider: Optional[Callable] = None
        self._news_provider: Optional[Callable] = None

    def _build_asset_list(self, config: Dict[str, Any]) -> Dict[str, AssetClass]:
        """Build flat asset -> asset_class mapping from config."""
        assets = {}
        asset_cfg = config.get("assets", {})
        for class_name, class_enum in [
            ("crypto", AssetClass.CRYPTO),
            ("stocks", AssetClass.STOCKS),
            ("forex", AssetClass.FOREX),
        ]:
            for symbol in asset_cfg.get(class_name, []):
                assets[symbol] = class_enum
        return assets

    def initialize_agents(self) -> Dict[str, bool]:
        """Initialize all sub-agents. Returns status per agent.

        Each agent is initialized independently so one failure
        does not prevent others from running.
        """
        status = {}

        # Technical Analysis Agent
        try:
            self._tech_agent = TechnicalAnalysisAgent(config_override=self.config)
            status["technical"] = True
            logger.info("Technical Analysis Agent initialized")
        except Exception as e:
            logger.error("Failed to initialize Technical Analysis Agent: %s", e)
            status["technical"] = False
            self._error_counts["init_technical"] += 1

        # Sentiment Analysis Agent
        try:
            sentiment_cfg = self.config.get("sentiment", {})
            social_cfg = sentiment_cfg.get("social", {})
            self._sentiment_agent = SentimentAnalysisAgent(
                subreddits=social_cfg.get("reddit_subreddits"),
            )
            status["sentiment"] = True
            logger.info("Sentiment Analysis Agent initialized")
        except Exception as e:
            logger.error("Failed to initialize Sentiment Analysis Agent: %s", e)
            status["sentiment"] = False
            self._error_counts["init_sentiment"] += 1

        # Risk Management subsystem
        try:
            self._portfolio_monitor = PortfolioMonitor(self.config)
            self._drawdown_manager = DrawdownManager(self.config)
            self._correlation_tracker = CorrelationTracker(self.config)
            self._risk_gate = RiskGate(
                portfolio_monitor=self._portfolio_monitor,
                drawdown_manager=self._drawdown_manager,
                correlation_tracker=self._correlation_tracker,
                config=self.config,
            )
            status["risk"] = True
            logger.info("Risk Management Agent initialized")
        except Exception as e:
            logger.error("Failed to initialize Risk Management Agent: %s", e)
            status["risk"] = False
            self._error_counts["init_risk"] += 1

        # Execution Agent
        try:
            self._execution_agent = ExecutionAgent(self.config)
            status["execution"] = True
            mode_str = "PAPER" if self._paper_mode else "LIVE"
            logger.info("Execution Agent initialized (%s mode)", mode_str)
        except Exception as e:
            logger.error("Failed to initialize Execution Agent: %s", e)
            status["execution"] = False
            self._error_counts["init_execution"] += 1

        # Trade Logger
        try:
            data_cfg = self.config.get("data", {})
            db_path = data_cfg.get("sqlite_path", "data/trading.db")
            self._trade_logger = TradeLogger(db_path=db_path)
            status["trade_logger"] = True
        except Exception as e:
            logger.error("Failed to initialize Trade Logger: %s", e)
            status["trade_logger"] = False

        # Telegram Alerts
        try:
            mon_cfg = self.config.get("monitoring", {})
            tg_cfg = mon_cfg.get("telegram", {})
            alert_cfg = mon_cfg.get("alerts", {})
            self._telegram = TelegramAlerts(
                bot_token=tg_cfg.get("bot_token", ""),
                chat_id=tg_cfg.get("chat_id", ""),
                alert_config=alert_cfg,
            )
            status["telegram"] = True
        except Exception as e:
            logger.error("Failed to initialize Telegram Alerts: %s", e)
            status["telegram"] = False

        return status

    def set_data_provider(self, provider: Callable) -> None:
        """Set callback to fetch market data for an asset.

        Provider signature: (asset: str, timeframe: str) -> pd.DataFrame
        Must return OHLCV DataFrame with DatetimeIndex.
        """
        self._data_provider = provider

    def set_news_provider(self, provider: Callable) -> None:
        """Set callback to fetch news for an asset.

        Provider signature: (asset: str) -> List[str]
        Returns list of news text strings.
        """
        self._news_provider = provider

    def set_portfolio_value(self, total_value: float, cash: float) -> None:
        """Set initial portfolio value for risk calculations."""
        if self._portfolio_monitor:
            self._portfolio_monitor.set_portfolio_value(total_value, cash)
        if self._drawdown_manager:
            self._drawdown_manager.initialize(total_value)
        self._circuit_breaker.reset_daily(total_value)

    async def run(self) -> None:
        """Start the main scan loop. Runs until stop() is called."""
        self._running = True
        mode_str = "PAPER" if self._paper_mode else "LIVE"
        logger.info(
            "Orchestrator starting scan loop: interval=%ds, mode=%s, "
            "confidence_threshold=%d%%, assets=%d",
            self.scan_interval, mode_str, self.min_confidence, len(self._assets),
        )

        while self._running:
            scan_start = time.monotonic()
            try:
                await self._run_scan_cycle()
            except Exception as e:
                logger.error(
                    "Scan cycle failed with unhandled error: %s\n%s",
                    e, traceback.format_exc(),
                )
                self._error_counts["scan_cycle"] += 1
                if self._telegram:
                    self._telegram.alert_system_error("orchestrator", str(e))

            # Sleep for remainder of interval
            elapsed = time.monotonic() - scan_start
            sleep_time = max(0, self.scan_interval - elapsed)
            if sleep_time > 0 and self._running:
                await asyncio.sleep(sleep_time)

        logger.info("Orchestrator scan loop stopped")

    def stop(self) -> None:
        """Signal the orchestrator to stop after the current scan cycle."""
        logger.info("Orchestrator stop requested")
        self._running = False

    async def _run_scan_cycle(self) -> None:
        """Execute one complete scan cycle across all assets."""
        self._scan_count += 1
        self._last_scan_time = datetime.utcnow()
        cycle_start = time.monotonic()

        logger.info("=== Scan cycle #%d started ===", self._scan_count)

        # Check circuit breakers first
        tripped, trip_reason = self._circuit_breaker.check()
        if tripped:
            logger.warning("Circuit breaker active: %s. Skipping scan.", trip_reason)
            return

        # Check drawdown pause
        if self._drawdown_manager:
            paused, pause_reason = self._drawdown_manager.check_pause_rules()
            if paused:
                logger.warning("Trading paused (drawdown): %s", pause_reason)
                return

        # Scan each asset
        signals_generated = 0
        trades_executed = 0

        for asset, asset_class in self._assets.items():
            try:
                result = await self._process_asset(asset, asset_class)
                if result.get("signal_generated"):
                    signals_generated += 1
                if result.get("trade_executed"):
                    trades_executed += 1
            except Exception as e:
                logger.error(
                    "Error processing asset %s: %s\n%s",
                    asset, e, traceback.format_exc(),
                )
                self._error_counts[f"asset_{asset}"] += 1

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "=== Scan cycle #%d complete: %.1fs, %d signals, %d trades ===",
            self._scan_count, elapsed, signals_generated, trades_executed,
        )

    async def _process_asset(
        self, asset: str, asset_class: AssetClass
    ) -> Dict[str, Any]:
        """Process a single asset: analyze, fuse signals, risk check, execute.

        Returns:
            Result dict with processing outcome.
        """
        result = {
            "asset": asset,
            "signal_generated": False,
            "trade_executed": False,
            "reason": "",
        }

        # -- Step 1: Collect Technical Signal --
        tech_signal = self._get_technical_signal(asset)

        # -- Step 2: Collect Sentiment Signal --
        sentiment_signal = self._get_sentiment_signal(asset)

        # -- Step 3: Fuse signals --
        fused_signal = self._fuse_signals(asset, tech_signal, sentiment_signal)

        if fused_signal is None or fused_signal.direction == Direction.HOLD:
            result["reason"] = "no_actionable_signal"
            return result

        result["signal_generated"] = True

        # -- Step 4: Check confidence threshold --
        if fused_signal.confidence < self.min_confidence:
            result["reason"] = (
                f"confidence_below_threshold: {fused_signal.confidence:.1f}% "
                f"< {self.min_confidence}%"
            )
            self._log_decision(asset, "SKIPPED", result["reason"], fused_signal)
            return result

        # -- Step 5: Build trade proposal --
        proposal = self._build_trade_proposal(asset, asset_class, fused_signal)
        if proposal is None:
            result["reason"] = "failed_to_build_proposal"
            return result

        # -- Step 6: Risk gate evaluation --
        risk_decision = self._evaluate_risk(proposal)
        if risk_decision is None:
            result["reason"] = "risk_agent_unavailable"
            return result

        if not risk_decision.approved:
            result["reason"] = f"risk_rejected: {risk_decision.reason}"
            self._log_decision(asset, "REJECTED", risk_decision.reason, fused_signal)
            return result

        # -- Step 7: Execute trade --
        trade_result = self._execute_trade(proposal, risk_decision)
        if trade_result is None:
            result["reason"] = "execution_agent_unavailable"
            return result

        if trade_result.status == TradeStatus.FILLED:
            result["trade_executed"] = True
            self._on_trade_executed(asset, trade_result, fused_signal)
        elif trade_result.status == TradeStatus.FAILED:
            result["reason"] = f"execution_failed: {trade_result.error}"
            self._circuit_breaker.record_trade(0.0, success=False)

        return result

    def _get_technical_signal(self, asset: str) -> Optional[Signal]:
        """Get technical analysis signal for an asset.

        Returns None if the agent is unavailable or analysis fails.
        """
        if self._tech_agent is None:
            return None

        if self._data_provider is None:
            logger.debug("No data provider set, skipping technical analysis for %s", asset)
            return None

        try:
            data = self._data_provider(asset, "1h")
            if data is None or len(data) < 100:
                logger.debug("Insufficient data for technical analysis: %s", asset)
                return None

            signal = self._tech_agent.analyze(asset, data, timeframe="1h")
            logger.debug(
                "Technical signal for %s: %s (conf=%.1f%%)",
                asset, signal.direction.value, signal.confidence,
            )
            return signal
        except Exception as e:
            logger.error("Technical analysis failed for %s: %s", asset, e)
            self._error_counts[f"tech_{asset}"] += 1
            return None

    def _get_sentiment_signal(self, asset: str) -> Optional[Signal]:
        """Get sentiment analysis signal for an asset.

        Returns None if the agent is unavailable or analysis fails.
        """
        if self._sentiment_agent is None:
            return None

        try:
            news_items = None
            if self._news_provider:
                news_items = self._news_provider(asset)

            signal = self._sentiment_agent.analyze(asset, news_items=news_items)
            logger.debug(
                "Sentiment signal for %s: %s (conf=%.1f%%)",
                asset, signal.direction.value, signal.confidence,
            )
            return signal
        except Exception as e:
            logger.error("Sentiment analysis failed for %s: %s", asset, e)
            self._error_counts[f"sentiment_{asset}"] += 1
            return None

    def _fuse_signals(
        self,
        asset: str,
        tech_signal: Optional[Signal],
        sentiment_signal: Optional[Signal],
    ) -> Optional[Signal]:
        """Fuse technical and sentiment signals using weighted combination.

        Default weights: 60% technical, 40% sentiment (configurable).
        If only one signal is available, it is used with reduced confidence.

        Args:
            asset: Asset symbol.
            tech_signal: Technical analysis signal (may be None).
            sentiment_signal: Sentiment analysis signal (may be None).

        Returns:
            Fused Signal, or None if no signals available.
        """
        tech_weight = self.signal_weights.get("technical", 0.60)
        sent_weight = self.signal_weights.get("sentiment", 0.40)

        if tech_signal is None and sentiment_signal is None:
            return None

        # Single signal available: use it with reduced confidence
        if tech_signal is None:
            return Signal(
                asset=asset,
                direction=sentiment_signal.direction,
                confidence=sentiment_signal.confidence * sent_weight * 0.7,  # single-signal penalty
                source="fused_sentiment_only",
                timeframe=sentiment_signal.timeframe,
                metadata={
                    "fusion_method": "single_source",
                    "sentiment_signal": {
                        "direction": sentiment_signal.direction.value,
                        "confidence": sentiment_signal.confidence,
                    },
                },
            )

        if sentiment_signal is None:
            return Signal(
                asset=asset,
                direction=tech_signal.direction,
                confidence=tech_signal.confidence * tech_weight * 0.7,  # single-signal penalty
                source="fused_technical_only",
                timeframe=tech_signal.timeframe,
                metadata={
                    "fusion_method": "single_source",
                    "technical_signal": {
                        "direction": tech_signal.direction.value,
                        "confidence": tech_signal.confidence,
                        "tier": tech_signal.metadata.get("signal_tier", "?"),
                    },
                },
            )

        # Both signals available: weighted fusion
        tech_dir = _direction_to_numeric(tech_signal.direction)
        sent_dir = _direction_to_numeric(sentiment_signal.direction)

        total_weight = tech_weight + sent_weight
        fused_direction = (tech_dir * tech_weight + sent_dir * sent_weight) / total_weight

        # Geometric mean for convergence (requires both signals to be strong)
        fused_confidence = (tech_signal.confidence ** tech_weight) * (sentiment_signal.confidence ** sent_weight) * 100

        # Agreement bonus: if both agree, boost confidence
        if (tech_dir > 0 and sent_dir > 0) or (tech_dir < 0 and sent_dir < 0):
            fused_confidence = min(100.0, fused_confidence * 1.35)
            agreement = "aligned"
        elif tech_dir == 0 or sent_dir == 0:
            agreement = "partial"
        else:
            # Disagreement: strong confidence penalty
            fused_confidence *= 0.50
            agreement = "conflicting"

        # Map fused direction back to Direction enum
        final_direction = _numeric_to_direction(fused_direction, fused_confidence)

        return Signal(
            asset=asset,
            direction=final_direction,
            confidence=min(100.0, max(0.0, fused_confidence)),
            source="fused",
            timeframe=tech_signal.timeframe,
            metadata={
                "fusion_method": "weighted_average",
                "tech_weight": tech_weight,
                "sent_weight": sent_weight,
                "agreement": agreement,
                "fused_direction_numeric": fused_direction,
                "technical_signal": {
                    "direction": tech_signal.direction.value,
                    "confidence": tech_signal.confidence,
                    "tier": tech_signal.metadata.get("signal_tier", "?"),
                },
                "sentiment_signal": {
                    "direction": sentiment_signal.direction.value,
                    "confidence": sentiment_signal.confidence,
                },
            },
        )

    def _calculate_recent_win_rate(self, lookback: int = 30) -> tuple[float, float, float]:
        """Calculate win rate, avg win, avg loss from recent closed trades."""
        try:
            recent_trades = self._trade_logger.get_recent_trades(lookback)
            if len(recent_trades) < 10:
                return 0.55, 0.02, 0.015  # fallback if insufficient data

            wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl', 0) <= 0]

            win_rate = len(wins) / len(recent_trades) if recent_trades else 0.55
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.02
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0.015

            # Normalize to percentages
            avg_win_pct = min(avg_win, 0.10)  # cap at 10%
            avg_loss_pct = min(avg_loss, 0.10)

            return win_rate, avg_win_pct, avg_loss_pct
        except Exception:
            return 0.55, 0.02, 0.015

    def _build_trade_proposal(
        self,
        asset: str,
        asset_class: AssetClass,
        signal: Signal,
    ) -> Optional[TradeProposal]:
        """Build a TradeProposal from a fused signal.

        Uses current market data for entry price, ATR-based stop/TP,
        and portfolio value for sizing.

        Returns None if insufficient data.
        """
        try:
            # Get current price from data provider
            entry_price = self._get_current_price(asset)
            if entry_price is None or entry_price <= 0:
                logger.warning("Cannot build proposal for %s: no price data", asset)
                return None

            # Get ATR for stop/TP calculation
            atr = self._get_current_atr(asset)
            if atr is None or atr <= 0:
                # Fallback: estimate ATR as 2% of price
                atr = entry_price * 0.02

            # Determine side
            side = "LONG" if signal.direction in (
                Direction.BUY, Direction.STRONG_BUY
            ) else "SHORT"

            # Calculate stop-loss and take-profit
            stop_loss = calculate_stop_loss(
                entry=entry_price, atr=atr, side=side, config=self.config,
            )
            tp_levels = calculate_take_profit(
                entry=entry_price, atr=atr, side=side, config=self.config,
            )
            take_profit = tp_levels[0] if tp_levels else entry_price

            # Calculate position size
            portfolio_value = self._get_portfolio_value()
            if portfolio_value <= 0:
                logger.warning("Portfolio value is zero, cannot size position")
                return None

            from src.risk.position_sizer import calculate_position_size
            streak = self._risk_gate.streak_info if self._risk_gate else {}

            win_rate, avg_win, avg_loss = self._calculate_recent_win_rate()

            position_value = calculate_position_size(
                portfolio_value=portfolio_value,
                signal_confidence=signal.confidence,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                config=self.config,
                consecutive_wins=streak.get("consecutive_wins", 0),
                consecutive_losses=streak.get("consecutive_losses", 0),
            )

            position_size = position_value / entry_price if entry_price > 0 else 0.0

            # Order type
            order_type_str = self.config.get("execution", {}).get(
                "default_order_type", "limit"
            )
            order_type_map = {
                "limit": OrderType.LIMIT,
                "market": OrderType.MARKET,
                "stop_limit": OrderType.STOP_LIMIT,
            }
            order_type = order_type_map.get(order_type_str, OrderType.LIMIT)

            proposal = TradeProposal(
                signal=signal,
                asset=asset,
                asset_class=asset_class,
                direction=signal.direction,
                entry_price=entry_price,
                position_size=position_size,
                position_value=position_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_type=order_type,
                confidence=signal.confidence,
                metadata={
                    "atr": atr,
                    "signal_source": signal.source,
                    "signal_metadata": signal.metadata,
                },
            )

            logger.info(
                "Trade proposal built: %s %s @ %.4f, size=%.4f ($%.2f), "
                "SL=%.4f, TP=%.4f, conf=%.1f%%",
                signal.direction.value, asset, entry_price,
                position_size, position_value, stop_loss, take_profit,
                signal.confidence,
            )
            return proposal

        except Exception as e:
            logger.error("Failed to build trade proposal for %s: %s", asset, e)
            return None

    def _evaluate_risk(self, proposal: TradeProposal) -> Optional[Any]:
        """Run the proposal through the risk gate.

        Returns RiskDecision or None if risk agent is unavailable.
        """
        if self._risk_gate is None:
            logger.warning("Risk gate not initialized, cannot evaluate proposal")
            return None

        try:
            portfolio = self._get_portfolio_state()
            decision = self._risk_gate.evaluate(proposal, portfolio)
            return decision
        except Exception as e:
            logger.error("Risk evaluation failed for %s: %s", proposal.asset, e)
            self._error_counts["risk_evaluation"] += 1
            return None

    def _execute_trade(
        self, proposal: TradeProposal, risk_decision: Any
    ) -> Optional[TradeResult]:
        """Execute a risk-approved trade.

        Returns TradeResult or None if execution agent is unavailable.
        """
        if self._execution_agent is None:
            logger.warning("Execution agent not initialized")
            return None

        try:
            result = self._execution_agent.execute(proposal, risk_decision)
            return result
        except Exception as e:
            logger.error("Trade execution failed for %s: %s", proposal.asset, e)
            self._error_counts["execution"] += 1
            return None

    def _on_trade_executed(
        self,
        asset: str,
        trade_result: TradeResult,
        signal: Signal,
    ) -> None:
        """Handle post-execution bookkeeping."""
        proposal = trade_result.proposal

        # Log the trade
        if self._trade_logger:
            try:
                self._trade_logger.log_trade(trade_result)
            except Exception as e:
                logger.error("Failed to log trade: %s", e)

        # Update circuit breaker
        self._circuit_breaker.record_trade(0.0, success=True)

        # Send telegram alert
        if self._telegram:
            side = "BUY" if proposal.direction in (
                Direction.BUY, Direction.STRONG_BUY
            ) else "SELL"
            self._telegram.alert_trade_executed(
                asset=asset,
                side=side,
                size=trade_result.fill_size,
                price=trade_result.fill_price,
                exchange=trade_result.exchange,
            )

        # Log decision
        self._log_decision(
            asset, "EXECUTED",
            f"Filled: {trade_result.fill_size:.6f} @ {trade_result.fill_price:.4f} "
            f"on {trade_result.exchange}",
            signal,
        )

        logger.info(
            "Trade executed: %s %s @ %.4f, size=%.6f, exchange=%s, slippage=%.4f%%",
            proposal.direction.value, asset, trade_result.fill_price,
            trade_result.fill_size, trade_result.exchange, trade_result.slippage,
        )

    def _get_current_price(self, asset: str) -> Optional[float]:
        """Get current price for an asset from the data provider."""
        if self._data_provider is None:
            return None
        try:
            data = self._data_provider(asset, "1h")
            if data is not None and len(data) > 0:
                return float(data["close"].iloc[-1])
        except Exception as e:
            logger.error("Failed to get price for %s: %s", asset, e)
        return None

    def _get_current_atr(self, asset: str) -> Optional[float]:
        """Get current ATR for an asset."""
        if self._data_provider is None:
            return None
        try:
            data = self._data_provider(asset, "1h")
            if data is not None and "atr" in data.columns and len(data) > 0:
                val = data["atr"].iloc[-1]
                if np.isfinite(val):
                    return float(val)
        except Exception:
            pass
        return None

    def _get_portfolio_value(self) -> float:
        """Get current portfolio total value."""
        if self._portfolio_monitor:
            state = self._portfolio_monitor.get_state()
            return state.total_value
        return 0.0

    def _get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        if self._portfolio_monitor:
            return self._portfolio_monitor.get_state()
        return PortfolioState(total_value=0.0, cash=0.0)

    def _log_decision(
        self,
        asset: str,
        action: str,
        reason: str,
        signal: Optional[Signal] = None,
    ) -> None:
        """Log a decision with full context."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "scan_cycle": self._scan_count,
            "asset": asset,
            "action": action,
            "reason": reason,
        }
        if signal:
            entry["signal"] = {
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "source": signal.source,
            }
        self._decision_log.append(entry)

        # Keep log bounded
        if len(self._decision_log) > 10000:
            self._decision_log = self._decision_log[-5000:]

    # --- Status and diagnostics ---

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get full orchestrator status for monitoring."""
        return {
            "running": self._running,
            "mode": "paper" if self._paper_mode else "live",
            "scan_count": self._scan_count,
            "scan_interval": self.scan_interval,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "min_confidence": self.min_confidence,
            "signal_weights": self.signal_weights,
            "assets_count": len(self._assets),
            "circuit_breaker": self._circuit_breaker.status,
            "error_counts": dict(self._error_counts),
            "agents": {
                "technical": self._tech_agent is not None,
                "sentiment": self._sentiment_agent is not None,
                "risk": self._risk_gate is not None,
                "execution": self._execution_agent is not None,
                "trade_logger": self._trade_logger is not None,
                "telegram": self._telegram is not None,
            },
        }

    @property
    def decision_log(self) -> List[Dict[str, Any]]:
        """Return recent decision log entries."""
        return list(self._decision_log[-100:])


# --- Helper functions ---

def _direction_to_numeric(direction: Direction) -> float:
    """Convert Direction enum to numeric value for signal fusion."""
    mapping = {
        Direction.STRONG_BUY: 1.0,
        Direction.BUY: 0.5,
        Direction.HOLD: 0.0,
        Direction.SELL: -0.5,
        Direction.STRONG_SELL: -1.0,
    }
    return mapping.get(direction, 0.0)


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
