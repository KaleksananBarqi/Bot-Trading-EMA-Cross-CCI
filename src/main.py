"""
Main Orchestrator — Entry point bot trading EMA Cross + CCI.

Menghubungkan semua komponen:
WebSocket → CandleManager → Indicators → Strategy → Risk → OrderManager

Lifecycle: Initialize → Start → Running → Graceful Shutdown
"""
from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import BotConfig, load_config
from src.data.candle_manager import CandleManager
from src.data.websocket_feed import WebSocketFeed
from src.strategy.ema_cci_strategy import (
    EmaCciStrategy, PositionState, SignalType, TradeSignal,
)
from src.risk.risk_manager import RiskManager, RiskParams
from src.execution.order_manager import OrderManager
from src.execution.position_tracker import PositionTracker
from src.notifications.telegram_notifier import TelegramNotifier
from src.database.mongo_manager import MongoManager
from src.utils.logger import log, setup_logger


class TradingBot:
    """
    Orchestrator utama bot trading.

    Mengelola lifecycle seluruh komponen dan mengkoordinasikan
    alur data dari WebSocket hingga eksekusi order.

    Attributes:
        config: Konfigurasi bot.
        candle_manager: Pengelola buffer candle.
        ws_feed: WebSocket feed dari Binance.
        strategy: Engine strategi EMA Cross + CCI.
        risk_manager: Pengelola risiko (SL, TP, sizing).
        order_manager: Eksekutor order.
        position_tracker: Pelacak posisi aktif.
        notifier: Notifikasi Telegram.
        db: MongoDB manager.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._running = False

        # Inisialisasi komponen
        self.position_tracker = PositionTracker()
        self.strategy = EmaCciStrategy(config)
        self.risk_manager = RiskManager(config)
        self.candle_manager = CandleManager(config)
        self.notifier = TelegramNotifier(config)
        self.db = MongoManager(config)

        # WebSocket & OrderManager diinisialisasi saat start
        self.ws_feed: WebSocketFeed | None = None
        self.order_manager: OrderManager | None = None

    async def start(self) -> None:
        """
        Inisialisasi dan mulai bot.

        Alur:
        1. Setup logger
        2. Connect MongoDB
        3. Init Telegram notifier
        4. Init WebSocket feed (+ fetch historical data)
        5. Register callback pada CandleManager
        6. Start WebSocket streaming
        """
        log.info("=" * 60)
        log.info("🚀 BOT EMA CROSS + CCI — Starting...")
        log.info("=" * 60)

        self._running = True

        try:
            # ── 1. MongoDB ──
            await self.db.connect()

            # ── 2. Telegram ──
            await self.notifier.initialize()

            # ── 3. WebSocket Feed ──
            self.ws_feed = WebSocketFeed(self.config, self.candle_manager)
            await self.ws_feed.initialize()

            # ── 4. Order Manager ──
            assert self.ws_feed.exchange is not None
            self.order_manager = OrderManager(
                config=self.config,
                exchange=self.ws_feed.exchange,
                position_tracker=self.position_tracker,
                notifier=self.notifier,
                db=self.db,
            )

            # ── 5. Register strategy callback ──
            self.candle_manager.register_callback(self._on_data_ready)

            # ── 6. Kirim notif startup ──
            await self.notifier.notify_startup(
                pairs=self.config.pair_symbols,
                timeframes=self.config.strategy.active_timeframes,
            )

            log.info(
                f"Bot siap — "
                f"Pairs: {self.config.pair_symbols} | "
                f"TF: {self.config.strategy.active_timeframes} | "
                f"Entry: {self.config.strategy.entry_mode} | "
                f"Testnet: {self.config.exchange.testnet}"
            )

            # ── 7. Set Margin & Leverage ──
            for pair_cfg in self.config.pairs:
                try:
                    await self.order_manager.exchange.set_margin_mode(pair_cfg.margin_mode, pair_cfg.symbol)
                    await self.order_manager.exchange.set_leverage(pair_cfg.leverage, pair_cfg.symbol)
                    log.info(f"Berhasil setup {pair_cfg.symbol}: mode {pair_cfg.margin_mode}, leverage {pair_cfg.leverage}x")
                except Exception as e:
                    log.error(f"Gagal mengatur leverage/margin untuk {pair_cfg.symbol}: {e}")

            # ── 8. Start streaming (blocking) ──
            await self.ws_feed.start()

        except asyncio.CancelledError:
            log.info("Bot dibatalkan, melakukan shutdown...")
        except Exception as e:
            log.error(f"Fatal error: {e}")
            if self.notifier:
                await self.notifier.notify_error(f"Fatal error: {e}")
            raise
        finally:
            await self.stop()

    async def _on_data_ready(self, pair: str, timeframe: str,
                             df: pd.DataFrame) -> None:
        """
        Callback saat data candle + indikator siap.
        Dipanggil oleh CandleManager setelah setiap candle close.

        Alur:
        1. Evaluasi strategi → sinyal?
        2. Jika ada sinyal → hitung risk
        3. Jika risk OK → eksekusi order

        Args:
            pair: Simbol pair.
            timeframe: Timeframe.
            df: DataFrame OHLCV + indikator.
        """
        try:
            # ── 1. Evaluasi strategi ──
            signal = self.strategy.evaluate(pair, timeframe, df)

            if signal is None:
                return  # Tidak ada sinyal

            # ── 2. Log sinyal ke MongoDB ──
            if self.db:
                await self.db.save_signal(signal, accepted=True)

            # ── 3. Cek posisi aktif ──
            if self.position_tracker.has_open_position(pair):
                log.info(
                    f"Sinyal {signal.signal_type.value} untuk {pair} "
                    f"diabaikan — sudah ada posisi aktif."
                )
                return

            # ── 4. Hitung risk parameters ──
            balance = await self._get_balance()
            pair_config = self.config.get_pair_config(pair)
            risk_params = self.risk_manager.calculate(signal, df, balance, pair_config)

            if risk_params is None:
                log.warning(f"Risk calculation gagal untuk {pair}, skip.")
                return

            # ── 5. Kirim notifikasi sinyal ──
            if self.notifier:
                await self.notifier.notify_signal(signal, risk_params)

            # ── 6. Eksekusi order ──
            assert self.order_manager is not None
            success = await self.order_manager.execute_signal(signal, risk_params)

            if success:
                # Update state machine
                state = (PositionState.LONG_SIGNAL
                         if signal.signal_type == SignalType.BUY
                         else PositionState.SHORT_SIGNAL)
                self.strategy.set_state(pair, timeframe, PositionState.IN_POSITION)

        except Exception as e:
            log.error(f"Error di _on_data_ready [{pair} {timeframe}]: {e}")
            if self.notifier:
                await self.notifier.notify_error(
                    f"Error processing {pair} [{timeframe}]: {e}"
                )

    async def _get_balance(self) -> float:
        """
        Ambil saldo akun dari exchange.

        Returns:
            Saldo USDT tersedia.
        """
        try:
            assert self.ws_feed is not None and self.ws_feed.exchange is not None
            balance = await self.ws_feed.exchange.fetch_balance()
            free_usdt = float(balance.get("free", {}).get("USDT", 0))
            log.debug(f"Balance USDT: ${free_usdt:,.2f}")
            return free_usdt
        except Exception as e:
            log.error(f"Gagal fetch balance: {e}")
            return 0.0

    async def stop(self) -> None:
        """Graceful shutdown semua komponen."""
        if not self._running:
            return

        self._running = False
        log.info("🔴 Shutting down bot...")

        # Kirim notif shutdown
        try:
            await self.notifier.notify_shutdown()
        except Exception:
            pass

        # Stop WebSocket
        if self.ws_feed:
            await self.ws_feed.stop()

        # Disconnect MongoDB
        await self.db.disconnect()

        # Print summary
        summary = self.position_tracker.get_summary()
        log.info(
            f"📊 Session Summary — "
            f"Trades: {summary['total_trades']} | "
            f"Wins: {summary['wins']} | "
            f"Losses: {summary['losses']} | "
            f"Win Rate: {summary['win_rate']:.1f}% | "
            f"Total PnL: {summary['total_pnl_pct']:+.2f}%"
        )

        log.info("Bot dihentikan. Goodbye! 👋")


def main() -> None:
    """Entry point utama — jalankan bot."""
    # Tentukan path config relatif ke CWD
    config_path = "config.yaml"
    env_path = ".env"

    try:
        # Load & validasi config
        config = load_config(config_path, env_path)

        # Setup logger dengan config
        setup_logger(
            level=config.logging.level,
            log_file=config.logging.file,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
        )

        # Buat instance bot
        bot = TradingBot(config)

        # Setup signal handlers untuk graceful shutdown
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _signal_handler() -> None:
            log.info("Menerima sinyal shutdown...")
            loop.create_task(bot.stop())

        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

        # Jalankan bot
        try:
            loop.run_until_complete(bot.start())
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt diterima, menghentikan bot...")
            loop.run_until_complete(bot.stop())
        finally:
            loop.close()

    except FileNotFoundError as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
