"""
Order Manager — Eksekusi order ke Binance Futures via ccxt.

Mendukung 2 mode entry:
- "close": Market order pada candle close
- "pullback": Limit order di harga EMA fast (dengan TTL)

SL dan TP dipasang sebagai server-side orders.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import ccxt.pro as ccxtpro

from src.config.settings import BotConfig
from src.strategy.ema_cci_strategy import SignalType, TradeSignal
from src.risk.risk_manager import RiskParams
from src.execution.position_tracker import PositionTracker, PositionSide
from src.utils.logger import log

if TYPE_CHECKING:
    from src.notifications.telegram_notifier import TelegramNotifier
    from src.database.mongo_manager import MongoManager


class OrderManager:
    """
    Mengelola eksekusi order ke exchange.

    Attributes:
        config: Konfigurasi bot.
        exchange: Instance exchange ccxt.pro.
        position_tracker: Tracker posisi aktif.
        notifier: Telegram notifier (opsional).
        db: MongoDB manager (opsional).
    """

    def __init__(self, config: BotConfig, exchange: ccxtpro.Exchange,
                 position_tracker: PositionTracker,
                 notifier: "TelegramNotifier | None" = None,
                 db: "MongoManager | None" = None) -> None:
        self.config = config
        self.exchange = exchange
        self.position_tracker = position_tracker
        self.notifier = notifier
        self.db = db

    async def execute_signal(self, signal: TradeSignal,
                             risk_params: RiskParams) -> bool:
        """
        Eksekusi sinyal trading: buka posisi + pasang SL/TP.

        Args:
            signal: Sinyal trading yang valid.
            risk_params: Parameter risiko (SL, TP, size).

        Returns:
            True jika eksekusi berhasil, False jika gagal.
        """
        pair = signal.pair

        # ── Guard: cek apakah sudah ada posisi aktif ──
        if self.position_tracker.has_open_position(pair):
            log.warning(f"Posisi sudah ada untuk {pair}, skip eksekusi.")
            return False

        try:
            entry_mode = self.config.strategy.entry_mode
            side = "buy" if signal.signal_type == SignalType.BUY else "sell"

            # ── 1. Buka posisi di tracker ──
            position = self.position_tracker.open_position(
                pair=pair,
                timeframe=signal.timeframe,
                signal_type=signal.signal_type,
                entry_price=signal.entry_price,
                risk_params=risk_params,
                cci_value=signal.cci_value,
                entry_mode=entry_mode,
            )

            # ── 2. Eksekusi Entry Order ──
            if entry_mode == "close":
                order = await self._market_order(pair, side, risk_params.position_size)
            else:  # pullback
                # Limit order di harga EMA fast
                limit_price = signal.ema_fast_value
                order = await self._limit_order(pair, side, risk_params.position_size,
                                                limit_price)

            if order is None:
                self.position_tracker.close_position(pair, signal.entry_price, "failed")
                return False

            order_id = order.get("id", "")
            self.position_tracker.update_order_ids(pair, order_id=order_id)

            # ── 3. Pasang Stop Loss (server-side) ──
            sl_order = await self._place_stop_loss(
                pair, signal.signal_type, risk_params.position_size,
                risk_params.stop_loss
            )
            if sl_order:
                self.position_tracker.update_order_ids(
                    pair, sl_order_id=sl_order.get("id", "")
                )

            # ── 4. Pasang Take Profit (server-side) ──
            tp_order = await self._place_take_profit(
                pair, signal.signal_type, risk_params.position_size,
                risk_params.take_profit
            )
            if tp_order:
                self.position_tracker.update_order_ids(
                    pair, tp_order_id=tp_order.get("id", "")
                )

            # ── 5. Kirim notifikasi Telegram ──
            if self.notifier and self.config.telegram.notify_fills:
                await self.notifier.notify_order_filled(signal, risk_params, order_id)

            # ── 6. Simpan ke MongoDB ──
            if self.db:
                await self.db.save_trade(position)

            log.info(
                f"✅ Order tereksekusi — {pair} [{signal.timeframe}] "
                f"{side.upper()} | Mode={entry_mode} | "
                f"OrderID={order_id}"
            )
            return True

        except Exception as e:
            log.error(f"❌ Gagal eksekusi order {pair}: {e}")

            # Rollback posisi di tracker
            self.position_tracker.close_position(pair, signal.entry_price, "error")

            # Kirim notifikasi error
            if self.notifier and self.config.telegram.notify_errors:
                await self.notifier.notify_error(
                    f"Gagal eksekusi order {pair}: {e}"
                )

            return False

    async def _market_order(self, pair: str, side: str,
                            amount: float) -> dict[str, Any] | None:
        """
        Eksekusi market order.

        Args:
            pair: Simbol pair.
            side: "buy" atau "sell".
            amount: Jumlah kontrak/unit.

        Returns:
            Order response dict, atau None jika gagal.
        """
        try:
            order = await self.exchange.create_order(
                symbol=pair,
                type="market",
                side=side,
                amount=amount,
            )
            log.info(
                f"Market order — {side.upper()} {amount} {pair} | "
                f"ID={order.get('id')}"
            )
            return order

        except Exception as e:
            log.error(f"Market order gagal {pair}: {e}")
            return None

    async def _limit_order(self, pair: str, side: str,
                           amount: float, price: float) -> dict[str, Any] | None:
        """
        Eksekusi limit order (untuk pullback mode).

        Args:
            pair: Simbol pair.
            side: "buy" atau "sell".
            amount: Jumlah kontrak/unit.
            price: Harga limit.

        Returns:
            Order response dict, atau None jika gagal.
        """
        try:
            order = await self.exchange.create_order(
                symbol=pair,
                type="limit",
                side=side,
                amount=amount,
                price=price,
            )
            log.info(
                f"Limit order — {side.upper()} {amount} {pair} @ {price} | "
                f"ID={order.get('id')}"
            )
            return order

        except Exception as e:
            log.error(f"Limit order gagal {pair}: {e}")
            return None

    async def _place_stop_loss(self, pair: str, signal_type: SignalType,
                               amount: float, sl_price: float) -> dict[str, Any] | None:
        """
        Pasang server-side stop loss order.

        Untuk BUY (long): SL = sell stop market
        Untuk SELL (short): SL = buy stop market
        """
        try:
            close_side = "sell" if signal_type == SignalType.BUY else "buy"
            order = await self.exchange.create_order(
                symbol=pair,
                type="stop_market",
                side=close_side,
                amount=amount,
                params={
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                },
            )
            log.info(
                f"SL order — {pair} @ {sl_price} | "
                f"Side={close_side} | ID={order.get('id')}"
            )
            return order

        except Exception as e:
            log.error(f"SL order gagal {pair} @ {sl_price}: {e}")
            return None

    async def _place_take_profit(self, pair: str, signal_type: SignalType,
                                 amount: float, tp_price: float) -> dict[str, Any] | None:
        """
        Pasang server-side take profit order.

        Untuk BUY (long): TP = sell take profit market
        Untuk SELL (short): TP = buy take profit market
        """
        try:
            close_side = "sell" if signal_type == SignalType.BUY else "buy"
            order = await self.exchange.create_order(
                symbol=pair,
                type="take_profit_market",
                side=close_side,
                amount=amount,
                params={
                    "stopPrice": tp_price,
                    "reduceOnly": True,
                },
            )
            log.info(
                f"TP order — {pair} @ {tp_price} | "
                f"Side={close_side} | ID={order.get('id')}"
            )
            return order

        except Exception as e:
            log.error(f"TP order gagal {pair} @ {tp_price}: {e}")
            return None

    async def cancel_all_orders(self, pair: str) -> None:
        """Batalkan semua open orders untuk pair tertentu."""
        try:
            await self.exchange.cancel_all_orders(pair)
            log.info(f"Semua order dibatalkan untuk {pair}")
        except Exception as e:
            log.error(f"Gagal membatalkan order {pair}: {e}")
