"""
Telegram Notifier — Kirim notifikasi sinyal, order, dan error ke Telegram.
Hanya notifikasi satu arah (bot → user), bukan command handler.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import Bot
from telegram.constants import ParseMode

from src.config.settings import BotConfig
from src.utils.logger import log

if TYPE_CHECKING:
    from src.strategy.ema_cci_strategy import TradeSignal
    from src.risk.risk_manager import RiskParams
    from src.execution.position_tracker import Position


class TelegramNotifier:
    """
    Mengirim notifikasi ke Telegram chat.

    Mendukung notifikasi untuk:
    - Sinyal baru (BUY/SELL)
    - Order filled
    - SL/TP hit
    - Error/disconnect

    Attributes:
        config: Konfigurasi bot.
        bot: Instance Telegram Bot.
        chat_id: Target chat ID.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.chat_id = config.telegram_chat_id
        self.thread_id = config.telegram_message_thread_id
        self.bot: Bot | None = None

    async def initialize(self) -> None:
        """Inisialisasi Telegram Bot."""
        if not self.config.telegram.enabled:
            log.info("Telegram notifier dinonaktifkan via config.")
            return

        if not self.config.telegram_bot_token:
            log.warning("Telegram bot token kosong, notifier dinonaktifkan.")
            return

        if not self.chat_id:
            log.warning("Telegram chat ID kosong, notifier dinonaktifkan.")
            return

        self.bot = Bot(token=self.config.telegram_bot_token)
        log.info("Telegram notifier diinisialisasi.")

    async def _send(self, message: str) -> None:
        """
        Kirim pesan ke Telegram.

        Args:
            message: Pesan dalam format Markdown.
        """
        if not self.bot or not self.config.telegram.enabled:
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                message_thread_id=self.thread_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.error(f"Gagal kirim notifikasi Telegram: {e}")

    async def notify_signal(self, signal: "TradeSignal",
                            risk_params: "RiskParams") -> None:
        """Kirim notifikasi saat sinyal baru terdeteksi."""
        if not self.config.telegram.notify_signals:
            return

        emoji = "🟢" if signal.signal_type.value == "BUY" else "🔴"
        side = signal.signal_type.value

        risk_pct = abs(signal.entry_price - risk_params.stop_loss) / signal.entry_price * 100
        reward_pct = abs(risk_params.take_profit - signal.entry_price) / signal.entry_price * 100

        msg = (
            f"{emoji} <b>SINYAL {side}</b> — {signal.pair} [{signal.timeframe}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Entry : <code>${signal.entry_price:,.2f}</code>\n"
            f"SL    : <code>${risk_params.stop_loss:,.2f}</code> (-{risk_pct:.2f}%)\n"
            f"TP    : <code>${risk_params.take_profit:,.2f}</code> (+{reward_pct:.2f}%)\n"
            f"R:R   : 1:{self.config.risk.risk_reward_ratio}\n"
            f"CCI   : <code>{signal.cci_value:+.1f}</code>\n"
            f"Mode  : {self.config.strategy.entry_mode}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def notify_order_filled(self, signal: "TradeSignal",
                                  risk_params: "RiskParams",
                                  order_id: str) -> None:
        """Kirim notifikasi saat order terisi."""
        if not self.config.telegram.notify_fills:
            return

        side = signal.signal_type.value
        msg = (
            f"✅ <b>ORDER FILLED</b> — {signal.pair}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Side  : {side}\n"
            f"Price : <code>${signal.entry_price:,.2f}</code>\n"
            f"Size  : <code>{risk_params.position_size:.4f}</code>\n"
            f"ID    : <code>{order_id}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def notify_position_closed(self, position: "Position") -> None:
        """Kirim notifikasi saat posisi ditutup (SL/TP hit)."""
        if not self.config.telegram.notify_sl_tp:
            return

        pnl = position.pnl or 0
        emoji = "🎯" if position.exit_type == "tp" else "🛑"
        result_emoji = "✅" if pnl > 0 else "❌"

        exit_type_label = {
            "tp": "Take Profit",
            "sl": "Stop Loss",
            "manual": "Manual Close",
        }.get(position.exit_type or "", position.exit_type or "Unknown")

        msg = (
            f"{emoji} <b>POSISI DITUTUP</b> — {position.pair}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Side   : {position.side.value.upper()}\n"
            f"Entry  : <code>${position.entry_price:,.2f}</code>\n"
            f"Exit   : <code>${position.exit_price or 0:,.2f}</code>\n"
            f"Type   : {exit_type_label}\n"
            f"PnL    : {result_emoji} <code>{pnl:+.2f}%</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def notify_error(self, error_message: str) -> None:
        """Kirim notifikasi saat terjadi error."""
        if not self.config.telegram.notify_errors:
            return

        msg = (
            f"⚠️ <b>ERROR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<code>{error_message[:500]}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def notify_startup(self, pairs: list[str],
                              timeframes: list[str]) -> None:
        """Kirim notifikasi saat bot dimulai."""
        pairs_str = ", ".join(pairs)
        tf_str = ", ".join(timeframes)

        msg = (
            f"🚀 <b>BOT STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategy : EMA Cross + CCI\n"
            f"Pairs    : {pairs_str}\n"
            f"TF       : {tf_str}\n"
            f"Mode     : {self.config.strategy.entry_mode}\n"
            f"Testnet  : {'Ya' if self.config.exchange.testnet else 'Tidak'}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        await self._send(msg)

    async def notify_shutdown(self) -> None:
        """Kirim notifikasi saat bot berhenti."""
        msg = "🔴 <b>BOT STOPPED</b>\nBot telah dihentikan."
        await self._send(msg)
