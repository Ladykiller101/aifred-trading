"""Main entry point for the multi-agent trading system.

Usage:
    python -m src.main                          # Paper trading, default config
    python -m src.main --mode paper             # Explicit paper mode
    python -m src.main --mode live              # Live trading (requires credentials)
    python -m src.main --config path/to/config.yaml
    python -m src.main --scan-interval 30       # 30-second scan interval
    python -m src.main --assets BTC/USDT,ETH/USDT  # Specific assets only

Features:
    - Parses command-line arguments
    - Loads configuration from YAML (with env var resolution)
    - Initializes all agents
    - Starts the orchestrator scan loop
    - Handles graceful shutdown (SIGINT/SIGTERM)
    - Logs startup/shutdown events
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import load_config, get_config
from src.data.market_data_provider import MarketDataProvider
from src.orchestrator import Orchestrator

logger = logging.getLogger("aifred")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AIFred Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                           # Paper trading with defaults
  python -m src.main --mode paper --scan-interval 30
  python -m src.main --mode live --config prod.yaml
  python -m src.main --assets BTC/USDT,ETH/USDT --scan-interval 15
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode: 'paper' (simulated) or 'live' (real orders). Default: paper",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Default: src/config/default.yaml",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=None,
        help="Scan interval in seconds. Overrides config value. Default: 60",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated list of assets to trade (overrides config). "
             "Example: BTC/USDT,ETH/USDT,SOL/USDT",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10000.0,
        help="Initial portfolio value in USD (for paper mode). Default: 10000",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level. Overrides config value.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path. Overrides config value.",
    )
    return parser.parse_args()


def setup_logging(config: dict, args: argparse.Namespace) -> None:
    """Configure logging based on config and CLI overrides."""
    mon_cfg = config.get("monitoring", {})
    log_cfg = mon_cfg.get("logging", {})

    level_str = args.log_level or log_cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    log_file = args.log_file or log_cfg.get("file", "logs/trading.log")

    # Create log directory
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning("Could not create log file %s: %s", log_file, e)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to the configuration dict."""
    # Mode override
    if args.mode:
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["mode"] = args.mode

    # Scan interval override
    if args.scan_interval is not None:
        if "orchestrator" not in config:
            config["orchestrator"] = {}
        config["orchestrator"]["scan_interval_seconds"] = args.scan_interval

    # Assets override
    if args.assets:
        asset_list = [a.strip() for a in args.assets.split(",") if a.strip()]
        if asset_list:
            # Classify assets by type
            crypto_assets = [a for a in asset_list if "/" in a]
            stock_assets = [a for a in asset_list if "/" not in a and len(a) <= 5]
            if "assets" not in config:
                config["assets"] = {}
            if crypto_assets:
                config["assets"]["crypto"] = crypto_assets
            else:
                config["assets"]["crypto"] = []
            if stock_assets:
                config["assets"]["stocks"] = stock_assets
            else:
                config["assets"]["stocks"] = []
            config["assets"]["forex"] = []

    return config


def print_startup_banner(config: dict, args: argparse.Namespace) -> None:
    """Print startup information."""
    mode = config.get("execution", {}).get("mode", "paper")
    scan_interval = config.get("orchestrator", {}).get("scan_interval_seconds", 60)
    confidence = config.get("orchestrator", {}).get("min_confidence_threshold", 70)

    asset_count = sum(
        len(config.get("assets", {}).get(k, []))
        for k in ("crypto", "stocks", "forex")
    )

    banner = f"""
================================================================================
  AIFred Multi-Agent Trading System
================================================================================
  Mode:             {mode.upper()}
  Scan Interval:    {scan_interval}s
  Confidence:       {confidence}%
  Assets:           {asset_count}
  Portfolio Value:  ${args.portfolio_value:,.2f}
  Config:           {args.config or 'default'}
  Started:          {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
================================================================================
"""
    print(banner)
    logger.info("AIFred starting: mode=%s, interval=%ds, assets=%d",
                mode, scan_interval, asset_count)


async def run_orchestrator(
    orchestrator: Orchestrator,
    shutdown_event: asyncio.Event,
) -> None:
    """Run the orchestrator until shutdown is signaled."""
    # Run orchestrator in a task
    orch_task = asyncio.create_task(orchestrator.run())

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Stop gracefully
    logger.info("Shutdown signal received, stopping orchestrator...")
    orchestrator.stop()

    # Wait for orchestrator to finish current cycle
    try:
        await asyncio.wait_for(orch_task, timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Orchestrator did not stop within 30s, cancelling")
        orch_task.cancel()
        try:
            await orch_task
        except asyncio.CancelledError:
            pass


def main() -> int:
    """Main entry point. Returns exit code."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(config_path=args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Setup logging
    setup_logging(config, args)

    # Print banner
    print_startup_banner(config, args)

    # Create orchestrator
    orchestrator = Orchestrator(config)

    # Initialize all agents
    logger.info("Initializing agents...")
    agent_status = orchestrator.initialize_agents()
    for agent_name, is_ok in agent_status.items():
        status_str = "OK" if is_ok else "FAILED"
        log_fn = logger.info if is_ok else logger.error
        log_fn("  Agent %-20s: %s", agent_name, status_str)

    # Check critical agents
    critical_agents = ["technical", "risk", "execution"]
    failed_critical = [a for a in critical_agents if not agent_status.get(a, False)]
    if failed_critical:
        logger.error(
            "Critical agents failed to initialize: %s. Aborting.",
            ", ".join(failed_critical),
        )
        return 1

    # Set initial portfolio value
    orchestrator.set_portfolio_value(
        total_value=args.portfolio_value,
        cash=args.portfolio_value,
    )
    logger.info("Portfolio initialized: $%.2f", args.portfolio_value)

    # Wire up market data provider
    data_cfg = config.get("data", {})
    market_data_provider = MarketDataProvider(
        default_exchange=data_cfg.get("default_exchange", "binance"),
        cache_ttl=data_cfg.get("cache_ttl_seconds", 60),
        min_candles=data_cfg.get("min_candles", 200),
    )
    orchestrator.set_data_provider(market_data_provider.get_data)
    logger.info("Market data provider initialized (exchange=%s)",
                data_cfg.get("default_exchange", "binance"))

    # Setup shutdown event
    shutdown_event = asyncio.Event()
    loop = asyncio.new_event_loop()

    def _signal_handler(sig, frame):
        sig_name = signal.Signals(sig).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        loop.call_soon_threadsafe(shutdown_event.set)

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    start_time = time.monotonic()
    exit_code = 0

    try:
        loop.run_until_complete(
            run_orchestrator(orchestrator, shutdown_event)
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        orchestrator.stop()
    except Exception as e:
        logger.error("Unhandled error in main loop: %s", e, exc_info=True)
        exit_code = 1
    finally:
        # Cleanup
        elapsed = time.monotonic() - start_time
        status = orchestrator.get_status()

        logger.info(
            "AIFred shutdown complete. Runtime: %.1fs, Scans: %d, Errors: %s",
            elapsed, status["scan_count"], status["error_counts"],
        )

        loop.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
