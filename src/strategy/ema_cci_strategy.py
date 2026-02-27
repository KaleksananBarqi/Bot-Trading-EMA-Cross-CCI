"""
Strategy Engine — Logika sinyal Buy/Sell berdasarkan EMA Cross + CCI Filter.

Aturan:
- BUY:  EMA fast cross above EMA slow + CCI > 0
- SELL: EMA fast cross below EMA slow + CCI < 0
- TIDAK BOLEH entry tanpa konfirmasi CCI (hard gate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from src.config.settings import BotConfig
from src.utils.logger import log

if TYPE_CHECKING:
    pass


class SignalType(str, Enum):
    """Tipe sinyal trading."""
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class PositionState(str, Enum):
    """State machine posisi per pair×timeframe."""
    IDLE = "IDLE"
    LONG_SIGNAL = "LONG_SIGNAL"
    SHORT_SIGNAL = "SHORT_SIGNAL"
    IN_POSITION = "IN_POSITION"


@dataclass
class TradeSignal:
    """Data sinyal trading yang dihasilkan oleh strategi."""
    signal_type: SignalType
    pair: str
    timeframe: str
    entry_price: float
    ema_fast_value: float
    ema_slow_value: float
    cci_value: float
    timestamp: pd.Timestamp | None = None


class EmaCciStrategy:
    """
    Otak strategi bot — mendeteksi sinyal BUY/SELL berdasarkan
    EMA crossover + CCI momentum filter.

    State machine mencegah sinyal duplikat:
    IDLE → LONG_SIGNAL/SHORT_SIGNAL → IN_POSITION → IDLE

    Attributes:
        config: Konfigurasi bot.
        states: Dict tracking state per (pair, timeframe).
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.states: dict[tuple[str, str], PositionState] = {}

    def get_state(self, pair: str, timeframe: str) -> PositionState:
        """Ambil state saat ini untuk pair×timeframe."""
        return self.states.get((pair, timeframe), PositionState.IDLE)

    def set_state(self, pair: str, timeframe: str, state: PositionState) -> None:
        """Set state untuk pair×timeframe."""
        old = self.states.get((pair, timeframe), PositionState.IDLE)
        self.states[(pair, timeframe)] = state
        if old != state:
            log.info(f"State {pair} [{timeframe}]: {old.value} → {state.value}")

    def evaluate(self, pair: str, timeframe: str,
                 df: pd.DataFrame) -> TradeSignal | None:
        """
        Evaluasi DataFrame terbaru untuk menghasilkan sinyal trading.

        Args:
            pair: Simbol pair.
            timeframe: Timeframe.
            df: DataFrame OHLCV + kolom indikator (ema_X, cci_X).

        Returns:
            TradeSignal jika ada sinyal valid, None jika tidak ada.
        """
        cfg = self.config.strategy
        ema_fast_col = f"ema_{cfg.ema_fast}"
        ema_slow_col = f"ema_{cfg.ema_slow}"
        cci_col = f"cci_{cfg.cci_length}"

        # Validasi kolom indikator ada
        required = {ema_fast_col, ema_slow_col, cci_col}
        missing = required - set(df.columns)
        if missing:
            log.warning(f"Kolom indikator tidak lengkap: {missing}")
            return None

        # Butuh minimal 2 candle untuk deteksi crossover
        if len(df) < 2:
            return None

        # Ambil 2 baris terakhir
        prev = df.iloc[-2]  # Candle sebelumnya
        curr = df.iloc[-1]  # Candle saat ini (baru ditutup)

        # Nilai indikator
        ema_fast_prev = prev[ema_fast_col]
        ema_slow_prev = prev[ema_slow_col]
        ema_fast_curr = curr[ema_fast_col]
        ema_slow_curr = curr[ema_slow_col]
        cci_curr = curr[cci_col]

        # Skip jika ada NaN (data belum cukup)
        values_to_check = [ema_fast_prev, ema_slow_prev,
                           ema_fast_curr, ema_slow_curr, cci_curr]
        if any(pd.isna(v) for v in values_to_check):
            return None

        # Cek state saat ini
        current_state = self.get_state(pair, timeframe)

        # Jika sudah dalam posisi, jangan generate sinyal baru
        if current_state == PositionState.IN_POSITION:
            return None

        # ──────────────────────────────────────────────
        # Deteksi Crossover
        # ──────────────────────────────────────────────

        # CROSS ABOVE: EMA fast menembus ke atas EMA slow
        cross_above = (ema_fast_prev <= ema_slow_prev) and (ema_fast_curr > ema_slow_curr)

        # CROSS BELOW: EMA fast menembus ke bawah EMA slow
        cross_below = (ema_fast_prev >= ema_slow_prev) and (ema_fast_curr < ema_slow_curr)

        signal: TradeSignal | None = None
        timestamp = curr.name if isinstance(curr.name, pd.Timestamp) else None

        # ──────────────────────────────────────────────
        # SINYAL BUY
        # ──────────────────────────────────────────────
        if cross_above:
            if cci_curr > 0:
                # ✅ TRIGGER + FILTER terpenuhi
                signal = TradeSignal(
                    signal_type=SignalType.BUY,
                    pair=pair,
                    timeframe=timeframe,
                    entry_price=float(curr["close"]),
                    ema_fast_value=float(ema_fast_curr),
                    ema_slow_value=float(ema_slow_curr),
                    cci_value=float(cci_curr),
                    timestamp=timestamp,
                )
                log.info(
                    f"🟢 SINYAL BUY — {pair} [{timeframe}] | "
                    f"Close={curr['close']:.4f} | "
                    f"EMA{cfg.ema_fast}={ema_fast_curr:.4f} > "
                    f"EMA{cfg.ema_slow}={ema_slow_curr:.4f} | "
                    f"CCI={cci_curr:.2f} > 0 ✅"
                )
            else:
                # ❌ CCI di bawah 0 → abaikan
                log.info(
                    f"⚠️ EMA Cross UP pada {pair} [{timeframe}] DIABAIKAN — "
                    f"CCI={cci_curr:.2f} < 0 (momentum lemah)"
                )

        # ──────────────────────────────────────────────
        # SINYAL SELL
        # ──────────────────────────────────────────────
        elif cross_below:
            if cci_curr < 0:
                # ✅ TRIGGER + FILTER terpenuhi
                signal = TradeSignal(
                    signal_type=SignalType.SELL,
                    pair=pair,
                    timeframe=timeframe,
                    entry_price=float(curr["close"]),
                    ema_fast_value=float(ema_fast_curr),
                    ema_slow_value=float(ema_slow_curr),
                    cci_value=float(cci_curr),
                    timestamp=timestamp,
                )
                log.info(
                    f"🔴 SINYAL SELL — {pair} [{timeframe}] | "
                    f"Close={curr['close']:.4f} | "
                    f"EMA{cfg.ema_fast}={ema_fast_curr:.4f} < "
                    f"EMA{cfg.ema_slow}={ema_slow_curr:.4f} | "
                    f"CCI={cci_curr:.2f} < 0 ✅"
                )
            else:
                # ❌ CCI di atas 0 → abaikan
                log.info(
                    f"⚠️ EMA Cross DOWN pada {pair} [{timeframe}] DIABAIKAN — "
                    f"CCI={cci_curr:.2f} > 0 (momentum lemah)"
                )

        return signal
