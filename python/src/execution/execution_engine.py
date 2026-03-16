"""Main execution engine: coordinates order routing, execution, and safety."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.types import (
    AssetClass, Direction, OrderType, Position, PortfolioState,
    RiskDecision, TradeProposal, TradeResult, TradeStatus,
)
from src.execution.exchange_connector import ExchangeConnector
from src.execution.order_manager import OrderManager, ManagedOrder
from src.execution.paper_trader import PaperTrader
from src.execution.safety_checks import SafetyChecks
from src.execution.smart_router import SmartRouter

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Main execution interface that coordinates all execution subsystems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        exec_config = config.get("execution", {})
        self._paper_mode = exec_config.get("mode", "paper") == "paper"

        # Initialize subsystems
        self.order_manager = OrderManager(
            max_retries=exec_config.get("max_consecutive_failures", 3)
        )
        self.safety = SafetyChecks(config)

        # Exchange connectors
        self._connectors: Dict[str, ExchangeConnector] = {}
        self._paper_trader: Optional[PaperTrader] = None
        self._router: Optional[SmartRouter] = None

        if self._paper_mode:
            self._paper_trader = PaperTrader(
                slippage_pct=exec_config.get("slippage_tolerance_pct", 0.1) / 2,
            )
            logger.info("Execution engine initialized in PAPER mode")
        else:
            self._init_connectors(config)
            logger.info("Execution engine initialized in LIVE mode")

        # Portfolio state (tracked internally)
        self._positions: Dict[str, Position] = {}

    def _init_connectors(self, config: Dict[str, Any]) -> None:
        """Initialize exchange connectors from config."""
        exchanges = config.get("exchanges", {})
        for asset_class_name, exchange_list in exchanges.items():
            for exch in exchange_list:
                name = exch.get("name", "")
                connector = ExchangeConnector(
                    name=name,
                    api_key=exch.get("api_key", ""),
                    secret=exch.get("secret", ""),
                    sandbox=False,
                    extra_params={k: v for k, v in exch.items()
                                  if k not in ("name", "api_key", "secret")},
                )
                connector.connect()
                self._connectors[name] = connector

        if self._connectors:
            self._router = SmartRouter(self._connectors)

    def is_paper_mode(self) -> bool:
        return self._paper_mode

    def execute(self, proposal: TradeProposal,
                risk_decision: RiskDecision) -> TradeResult:
        """Execute a trade proposal that has been approved by the risk gate.

        This is the main entry point for trade execution.
        """
        portfolio = self._get_portfolio_state()

        # Pre-execution safety checks
        passed, reason = self.safety.pre_execution_check(
            proposal, risk_decision, portfolio,
            self._paper_trader if self._paper_mode else None,
        )
        if not passed:
            logger.info("Trade rejected by safety checks: %s", reason)
            return TradeResult(
                proposal=proposal,
                status=TradeStatus.REJECTED,
                error=reason,
            )

        # Apply risk adjustments
        size = risk_decision.adjusted_size or proposal.position_size
        stop = risk_decision.adjusted_stop or proposal.stop_loss

        # Determine side
        side = "buy" if proposal.direction in (Direction.BUY, Direction.STRONG_BUY) else "sell"

        # Execute
        if self._paper_mode:
            result = self._execute_paper(proposal, side, size, stop)
        else:
            result = self._execute_live(proposal, side, size, stop)

        # Post-execution checks
        post_ok, post_reason = self.safety.post_execution_check(result, proposal)
        if not post_ok and result.status not in (TradeStatus.FILLED, TradeStatus.PARTIAL):
            logger.warning("Post-execution check flagged: %s", post_reason)

        # Place stop-loss if needed
        if result.status == TradeStatus.FILLED and self.safety.should_place_stop_loss(proposal):
            self._place_stop_loss(proposal, stop, size, side)

        # Track position
        if result.status == TradeStatus.FILLED:
            self._track_position(proposal, result, side, stop)

        return result

    def _execute_paper(self, proposal: TradeProposal, side: str,
                       size: float, stop: float) -> TradeResult:
        """Execute via paper trader."""
        assert self._paper_trader is not None
        self._paper_trader.set_price(proposal.asset, proposal.entry_price)

        order_type = proposal.order_type.value
        order = self._paper_trader.place_order(
            symbol=proposal.asset,
            side=side,
            order_type=order_type,
            amount=size,
            price=proposal.entry_price,
        )

        if order.get("status") == "rejected":
            return TradeResult(
                proposal=proposal,
                status=TradeStatus.FAILED,
                error=order.get("info", "paper_trade_rejected"),
            )

        fill_price = float(order.get("average", proposal.entry_price))
        slippage = abs(fill_price - proposal.entry_price) / proposal.entry_price * 100 if proposal.entry_price > 0 else 0
        fee_info = order.get("fee", {})

        return TradeResult(
            proposal=proposal,
            status=TradeStatus.FILLED,
            fill_price=fill_price,
            fill_size=size,
            slippage=slippage,
            fees=float(fee_info.get("cost", 0)),
            exchange="paper",
            order_id=order.get("id", ""),
        )

    def _execute_live(self, proposal: TradeProposal, side: str,
                      size: float, stop: float) -> TradeResult:
        """Execute on a real exchange via smart routing."""
        # Route to best exchange
        exchange_name = None
        if self._router:
            exchange_name = self._router.route_order(proposal.asset, side, size)

        if exchange_name is None:
            # Fall back to first available connector
            if not self._connectors:
                return TradeResult(
                    proposal=proposal, status=TradeStatus.FAILED,
                    error="no_exchange_available",
                )
            exchange_name = next(iter(self._connectors))

        connector = self._connectors[exchange_name]
        order_type = self._map_order_type(proposal.order_type)

        # Create and submit managed order
        managed = self.order_manager.create_order(
            symbol=proposal.asset, side=side,
            order_type=proposal.order_type,
            amount=size, price=proposal.entry_price,
        )

        try:
            result = self.order_manager.submit_order(managed, connector)
        except Exception as e:
            return TradeResult(
                proposal=proposal, status=TradeStatus.FAILED,
                error=str(e), exchange=exchange_name,
            )

        # Update status from exchange
        self.order_manager.update_order_status(managed, connector)

        # Record fill rate for smart router
        if self._router and managed.filled_amount > 0:
            self._router.record_fill(exchange_name, size, managed.filled_amount)

        status = TradeStatus.FILLED if managed.state.value == "filled" else TradeStatus.PARTIAL
        if managed.state.value in ("failed", "cancelled"):
            status = TradeStatus.FAILED

        slippage = 0.0
        if proposal.entry_price > 0 and managed.average_fill_price > 0:
            slippage = abs(managed.average_fill_price - proposal.entry_price) / proposal.entry_price * 100

        return TradeResult(
            proposal=proposal,
            status=status,
            fill_price=managed.average_fill_price,
            fill_size=managed.filled_amount,
            slippage=slippage,
            fees=managed.fees,
            exchange=exchange_name,
            order_id=managed.exchange_order_id or managed.id,
        )

    def _place_stop_loss(self, proposal: TradeProposal, stop: float,
                         size: float, entry_side: str) -> None:
        """Place a stop-loss order immediately after fill."""
        sl_side = "sell" if entry_side == "buy" else "buy"
        if self._paper_mode:
            logger.info("[PAPER] Stop-loss set for %s at %.4f", proposal.asset, stop)
            return

        # Place on the same exchange
        for name, connector in self._connectors.items():
            try:
                connector.place_order(
                    symbol=proposal.asset, side=sl_side,
                    order_type="stop", amount=size, price=stop,
                    params={"stopPrice": stop},
                )
                logger.info("Stop-loss placed for %s at %.4f on %s",
                            proposal.asset, stop, name)
                return
            except Exception as e:
                logger.error("Failed to place stop-loss on %s: %s", name, e)

    def _track_position(self, proposal: TradeProposal, result: TradeResult,
                        side: str, stop: float) -> None:
        """Track a new open position."""
        pos = Position(
            asset=proposal.asset,
            asset_class=proposal.asset_class,
            side="LONG" if side == "buy" else "SHORT",
            entry_price=result.fill_price,
            current_price=result.fill_price,
            size=result.fill_size,
            stop_loss=stop,
            take_profit=proposal.take_profit,
            order_id=result.order_id,
            strategy=proposal.signal.source if proposal.signal else "",
        )
        self._positions[proposal.asset] = pos
        if self._paper_mode and self._paper_trader:
            self._paper_trader.open_position(
                asset=proposal.asset, asset_class=proposal.asset_class,
                side=pos.side, entry_price=result.fill_price,
                size=result.fill_size, stop_loss=stop,
                take_profit=proposal.take_profit, order_id=result.order_id,
            )

    def close_position(self, position: Position, reason: str = "") -> TradeResult:
        """Close an open position."""
        side = "sell" if position.side == "LONG" else "buy"

        if self._paper_mode:
            assert self._paper_trader is not None
            self._paper_trader.set_price(position.asset, position.current_price)
            order = self._paper_trader.place_order(
                symbol=position.asset, side=side,
                order_type="market", amount=position.size,
            )
            pnl = self._paper_trader.close_position(position.asset, position.current_price)
            self._positions.pop(position.asset, None)
            fill_price = float(order.get("average", position.current_price))

            # Build a minimal proposal for the result
            from src.utils.types import Signal
            dummy_signal = Signal(
                asset=position.asset,
                direction=Direction.SELL if position.side == "LONG" else Direction.BUY,
                confidence=100.0, source="close:" + reason,
            )
            dummy_proposal = TradeProposal(
                signal=dummy_signal, asset=position.asset,
                asset_class=position.asset_class,
                direction=dummy_signal.direction,
                entry_price=position.current_price,
                position_size=position.size,
                position_value=position.size * position.current_price,
                stop_loss=0, take_profit=0,
            )
            return TradeResult(
                proposal=dummy_proposal, status=TradeStatus.FILLED,
                fill_price=fill_price, fill_size=position.size,
                exchange="paper", order_id=order.get("id", ""),
            )
        else:
            # Live close
            for name, connector in self._connectors.items():
                try:
                    result = connector.place_order(
                        symbol=position.asset, side=side,
                        order_type="market", amount=position.size,
                    )
                    self._positions.pop(position.asset, None)
                    from src.utils.types import Signal
                    dummy_signal = Signal(
                        asset=position.asset,
                        direction=Direction.SELL if position.side == "LONG" else Direction.BUY,
                        confidence=100.0, source="close:" + reason,
                    )
                    dummy_proposal = TradeProposal(
                        signal=dummy_signal, asset=position.asset,
                        asset_class=position.asset_class,
                        direction=dummy_signal.direction,
                        entry_price=position.current_price,
                        position_size=position.size,
                        position_value=position.size * position.current_price,
                        stop_loss=0, take_profit=0,
                    )
                    fill_price = float(result.get("average", position.current_price) or position.current_price)
                    return TradeResult(
                        proposal=dummy_proposal, status=TradeStatus.FILLED,
                        fill_price=fill_price, fill_size=position.size,
                        exchange=name, order_id=result.get("id", ""),
                    )
                except Exception as e:
                    logger.error("Failed to close position on %s: %s", name, e)

            from src.utils.types import Signal
            dummy_signal = Signal(
                asset=position.asset, direction=Direction.HOLD,
                confidence=0, source="close_failed",
            )
            dummy_proposal = TradeProposal(
                signal=dummy_signal, asset=position.asset,
                asset_class=position.asset_class, direction=Direction.HOLD,
                entry_price=0, position_size=0, position_value=0,
                stop_loss=0, take_profit=0,
            )
            return TradeResult(
                proposal=dummy_proposal, status=TradeStatus.FAILED,
                error="failed_to_close_on_all_exchanges",
            )

    def modify_stop(self, position: Position, new_stop: float) -> bool:
        """Modify the stop-loss for an open position."""
        if position.asset not in self._positions:
            logger.warning("Position %s not found", position.asset)
            return False
        self._positions[position.asset].stop_loss = new_stop
        logger.info("Stop-loss for %s updated to %.4f", position.asset, new_stop)
        return True

    def get_open_orders(self) -> List[ManagedOrder]:
        return self.order_manager.get_open_orders()

    def get_open_positions(self) -> List[Position]:
        return list(self._positions.values())

    def _get_portfolio_state(self) -> PortfolioState:
        """Build current portfolio state."""
        positions = list(self._positions.values())
        if self._paper_mode and self._paper_trader:
            total_value = self._paper_trader.get_total_value()
            balances = self._paper_trader.get_balance()
            cash = balances.get("free", {}).get("USD", 0) + balances.get("free", {}).get("USDT", 0)
        else:
            total_value = sum(p.current_price * p.size for p in positions)
            cash = 0.0
            for connector in self._connectors.values():
                try:
                    bal = connector.get_balance()
                    cash += float(bal.get("free", {}).get("USD", 0) or 0)
                    cash += float(bal.get("free", {}).get("USDT", 0) or 0)
                except Exception:
                    pass
            total_value += cash

        return PortfolioState(
            total_value=total_value,
            cash=cash,
            positions=positions,
        )

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        return {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LIMIT: "stop",
            OrderType.TRAILING_STOP: "trailing_stop",
        }.get(order_type, "limit")
