"""
Position Tracker — Tracking posisi aktif per pair×timeframe.
Memastikan hanya 1 posisi per pair per waktu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.strategy.ema_cci_strategy import SignalType
from src.risk.risk_manager import RiskParams
from src.utils.logger import log


class PositionSide(str, Enum):
    """Sisi posisi."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Status posisi."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """Data posisi trading aktif."""
    pair: str
    timeframe: str
    side: PositionSide
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""
    status: PositionStatus = PositionStatus.PENDING
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_type: str | None = None  # "tp", "sl", "manual"
    pnl: float | None = None
    cci_at_entry: float = 0.0
    entry_mode: str = "close"
    metadata: dict[str, Any] = field(default_factory=dict)


class PositionTracker:
    """
    Mengelola dan melacak semua posisi aktif.

    Aturan utama: Hanya SATU posisi per pair pada satu waktu.

    Attributes:
        positions: Dict mapping pair → Position aktif.
        closed_positions: List posisi yang sudah ditutup.
    """

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}  # key = pair
        self.closed_positions: list[Position] = []

    def has_open_position(self, pair: str) -> bool:
        """Cek apakah ada posisi aktif untuk pair tertentu."""
        pos = self.positions.get(pair)
        return pos is not None and pos.status in (PositionStatus.OPEN, PositionStatus.PENDING)

    def get_position(self, pair: str) -> Position | None:
        """Ambil posisi aktif untuk pair tertentu."""
        return self.positions.get(pair)

    def open_position(self, pair: str, timeframe: str,
                      signal_type: SignalType, entry_price: float,
                      risk_params: RiskParams, cci_value: float,
                      entry_mode: str) -> Position:
        """
        Buka posisi baru.

        Args:
            pair: Simbol pair.
            timeframe: Timeframe.
            signal_type: BUY atau SELL.
            entry_price: Harga entry.
            risk_params: Parameter risiko (SL, TP, size).
            cci_value: Nilai CCI saat entry.
            entry_mode: Mode entry ("close" atau "pullback").

        Returns:
            Position yang baru dibuat.

        Raises:
            ValueError: Jika sudah ada posisi aktif untuk pair ini.
        """
        if self.has_open_position(pair):
            raise ValueError(
                f"Sudah ada posisi aktif untuk {pair}. "
                f"Tutup posisi lama sebelum membuka yang baru."
            )

        side = PositionSide.LONG if signal_type == SignalType.BUY else PositionSide.SHORT

        position = Position(
            pair=pair,
            timeframe=timeframe,
            side=side,
            entry_price=entry_price,
            stop_loss=risk_params.stop_loss,
            take_profit=risk_params.take_profit,
            position_size=risk_params.position_size,
            cci_at_entry=cci_value,
            entry_mode=entry_mode,
        )

        self.positions[pair] = position

        log.info(
            f"📌 Posisi dibuka — {pair} [{timeframe}] "
            f"{side.value.upper()} | "
            f"Entry={entry_price:.4f} | "
            f"SL={risk_params.stop_loss:.4f} | "
            f"TP={risk_params.take_profit:.4f} | "
            f"Size={risk_params.position_size:.4f}"
        )

        return position

    def update_order_ids(self, pair: str, order_id: str = "",
                         sl_order_id: str = "", tp_order_id: str = "") -> None:
        """Update order IDs setelah order tereksekusi."""
        pos = self.positions.get(pair)
        if pos is None:
            return

        if order_id:
            pos.order_id = order_id
        if sl_order_id:
            pos.sl_order_id = sl_order_id
        if tp_order_id:
            pos.tp_order_id = tp_order_id

        pos.status = PositionStatus.OPEN
        log.debug(f"Order IDs diperbarui untuk {pair}: entry={order_id}")

    def close_position(self, pair: str, exit_price: float,
                       exit_type: str) -> Position | None:
        """
        Tutup posisi aktif.

        Args:
            pair: Simbol pair.
            exit_price: Harga exit.
            exit_type: Tipe exit ("tp", "sl", "manual").

        Returns:
            Position yang ditutup, atau None jika tidak ada posisi.
        """
        pos = self.positions.pop(pair, None)
        if pos is None:
            log.warning(f"Tidak ada posisi untuk ditutup: {pair}")
            return None

        pos.exit_price = exit_price
        pos.exit_time = datetime.now(timezone.utc)
        pos.exit_type = exit_type
        pos.status = PositionStatus.CLOSED

        # Hitung PnL
        if pos.side == PositionSide.LONG:
            pos.pnl = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pos.pnl = (pos.entry_price - exit_price) / pos.entry_price * 100

        self.closed_positions.append(pos)

        emoji = "✅" if (pos.pnl or 0) > 0 else "❌"
        log.info(
            f"{emoji} Posisi ditutup — {pair} [{pos.timeframe}] "
            f"{pos.side.value.upper()} | "
            f"Entry={pos.entry_price:.4f} → Exit={exit_price:.4f} | "
            f"PnL={pos.pnl:+.2f}% | "
            f"Exit={exit_type}"
        )

        return pos

    def get_all_open(self) -> list[Position]:
        """Ambil semua posisi yang sedang aktif."""
        return [
            pos for pos in self.positions.values()
            if pos.status in (PositionStatus.OPEN, PositionStatus.PENDING)
        ]

    def get_summary(self) -> dict:
        """Ringkasan statistik posisi."""
        total_closed = len(self.closed_positions)
        wins = sum(1 for p in self.closed_positions if (p.pnl or 0) > 0)
        losses = total_closed - wins
        total_pnl = sum(p.pnl or 0 for p in self.closed_positions)

        return {
            "open_positions": len(self.get_all_open()),
            "total_trades": total_closed,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total_closed * 100) if total_closed > 0 else 0,
            "total_pnl_pct": round(total_pnl, 4),
        }
