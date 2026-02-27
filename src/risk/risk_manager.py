"""
Risk Manager — Kalkulasi Stop Loss, Take Profit, dan Position Sizing.

Mendukung 2 mode SL:
- "swing_low": Berdasarkan swing high/low terdekat
- "ema_slow": Berdasarkan nilai EMA slow

Take Profit selalu R:R 1:2 (configurable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.settings import BotConfig
from src.strategy.ema_cci_strategy import SignalType, TradeSignal
from src.utils.logger import log


@dataclass
class RiskParams:
    """Parameter risiko yang sudah dihitung untuk satu trade."""
    stop_loss: float
    take_profit: float
    risk_distance: float
    position_size: float  # Dalam unit base currency
    risk_pct: float       # Persentase risiko


class RiskManager:
    """
    Menghitung SL, TP, dan position sizing berdasarkan konfigurasi risk.

    Attributes:
        config: Konfigurasi bot.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def calculate(self, signal: TradeSignal, df: pd.DataFrame,
                  balance: float) -> RiskParams | None:
        """
        Hitung parameter risiko untuk sinyal trading.

        Args:
            signal: Sinyal trading (BUY/SELL).
            df: DataFrame OHLCV + indikator.
            balance: Saldo akun saat ini (dalam USDT).

        Returns:
            RiskParams jika perhitungan berhasil, None jika gagal.
        """
        try:
            entry_price = signal.entry_price

            # Hitung Stop Loss
            sl = self._calculate_stop_loss(signal, df)
            if sl is None:
                log.warning(f"Gagal menghitung SL untuk {signal.pair}")
                return None

            # Hitung Risk Distance
            if signal.signal_type == SignalType.BUY:
                risk_distance = entry_price - sl
            else:  # SELL
                risk_distance = sl - entry_price

            # Validasi: risk distance harus positif
            if risk_distance <= 0:
                log.warning(
                    f"Risk distance tidak valid ({risk_distance:.4f}) "
                    f"untuk {signal.pair} [{signal.timeframe}] — "
                    f"entry={entry_price}, SL={sl}"
                )
                return None

            # Hitung Take Profit (R:R ratio)
            rr_ratio = self.config.risk.risk_reward_ratio
            if signal.signal_type == SignalType.BUY:
                tp = entry_price + (risk_distance * rr_ratio)
            else:  # SELL
                tp = entry_price - (risk_distance * rr_ratio)

            # Hitung Position Size
            max_risk_amount = balance * (self.config.risk.max_position_size_pct / 100)
            position_size = max_risk_amount / risk_distance

            risk_pct = (risk_distance / entry_price) * 100

            params = RiskParams(
                stop_loss=round(sl, 8),
                take_profit=round(tp, 8),
                risk_distance=round(risk_distance, 8),
                position_size=round(position_size, 8),
                risk_pct=round(risk_pct, 4),
            )

            log.info(
                f"📊 Risk — {signal.pair} [{signal.timeframe}] "
                f"{signal.signal_type.value} | "
                f"Entry={entry_price:.4f} | "
                f"SL={params.stop_loss:.4f} ({params.risk_pct:.2f}%) | "
                f"TP={params.take_profit:.4f} | "
                f"R:R=1:{rr_ratio} | "
                f"Size={params.position_size:.4f}"
            )

            return params

        except Exception as e:
            log.error(f"Error menghitung risk untuk {signal.pair}: {e}")
            return None

    def _calculate_stop_loss(self, signal: TradeSignal,
                             df: pd.DataFrame) -> float | None:
        """
        Hitung Stop Loss berdasarkan mode yang dikonfigurasi.

        Args:
            signal: Sinyal trading.
            df: DataFrame OHLCV + indikator.

        Returns:
            Harga SL, atau None jika gagal.
        """
        sl_mode = self.config.risk.sl_mode
        buffer_pct = self.config.risk.sl_buffer_pct / 100  # Convert ke desimal

        if sl_mode == "swing_low":
            return self._sl_swing_point(signal, df, buffer_pct)
        elif sl_mode == "ema_slow":
            return self._sl_ema_slow(signal, df, buffer_pct)
        else:
            log.error(f"SL mode tidak dikenal: {sl_mode}")
            return None

    def _sl_swing_point(self, signal: TradeSignal, df: pd.DataFrame,
                        buffer_pct: float) -> float | None:
        """
        SL berdasarkan swing high/low terdekat.

        Untuk BUY: SL di bawah swing low terdekat.
        Untuk SELL: SL di atas swing high terdekat.
        """
        lookback = self.config.strategy.swing_lookback

        if len(df) < lookback:
            log.warning(
                f"Data tidak cukup untuk swing detection: "
                f"butuh {lookback}, tersedia {len(df)}"
            )
            # Fallback ke EMA slow
            return self._sl_ema_slow(signal, df, buffer_pct)

        # Ambil N candle terakhir untuk deteksi swing
        recent = df.tail(lookback)

        if signal.signal_type == SignalType.BUY:
            # Cari swing low terdekat
            swing_low = self._find_swing_low(recent)
            if swing_low is None:
                swing_low = float(recent["low"].min())
            sl = swing_low * (1 - buffer_pct)

        else:  # SELL
            # Cari swing high terdekat
            swing_high = self._find_swing_high(recent)
            if swing_high is None:
                swing_high = float(recent["high"].max())
            sl = swing_high * (1 + buffer_pct)

        return sl

    def _sl_ema_slow(self, signal: TradeSignal, df: pd.DataFrame,
                     buffer_pct: float) -> float | None:
        """
        SL berdasarkan EMA slow ± buffer.

        Untuk BUY: SL di bawah EMA slow.
        Untuk SELL: SL di atas EMA slow.
        """
        ema_col = f"ema_{self.config.strategy.ema_slow}"
        if ema_col not in df.columns:
            log.error(f"Kolom {ema_col} tidak ditemukan")
            return None

        ema_value = float(df[ema_col].iloc[-1])
        if pd.isna(ema_value):
            log.error(f"Nilai {ema_col} adalah NaN")
            return None

        if signal.signal_type == SignalType.BUY:
            sl = ema_value * (1 - buffer_pct)
        else:  # SELL
            sl = ema_value * (1 + buffer_pct)

        return sl

    @staticmethod
    def _find_swing_low(df: pd.DataFrame) -> float | None:
        """
        Cari swing low (local minimum) dalam DataFrame.
        Swing low = candle yang low-nya lebih rendah dari candle sebelum dan sesudahnya.
        """
        lows = df["low"].values
        swing_lows = []

        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])

        if not swing_lows:
            return None

        # Ambil swing low terdekat (terakhir)
        return float(swing_lows[-1])

    @staticmethod
    def _find_swing_high(df: pd.DataFrame) -> float | None:
        """
        Cari swing high (local maximum) dalam DataFrame.
        Swing high = candle yang high-nya lebih tinggi dari candle sebelum dan sesudahnya.
        """
        highs = df["high"].values
        swing_highs = []

        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(highs[i])

        if not swing_highs:
            return None

        # Ambil swing high terdekat (terakhir)
        return float(swing_highs[-1])
