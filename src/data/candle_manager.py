"""
Candle Manager — Buffer dan manajemen DataFrame OHLCV.
Menyimpan rolling buffer candle per (pair, timeframe) dan
men-trigger perhitungan indikator saat candle baru ditutup.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from src.indicators.ema import add_ema_columns
from src.indicators.cci import add_cci_column
from src.config.settings import BotConfig
from src.utils.logger import log


class CandleManager:
    """
    Mengelola buffer candle OHLCV untuk setiap pair × timeframe.

    Attributes:
        config: Konfigurasi bot.
        buffer: Dict mapping (pair, timeframe) → pd.DataFrame.
        max_candles: Jumlah maksimal candle yang disimpan di buffer.
        on_data_ready: Callback yang dipanggil saat data indikator siap.
    """

    def __init__(self, config: BotConfig, max_candles: int = 100) -> None:
        self.config = config
        self.max_candles = max_candles
        self.buffer: dict[tuple[str, str], pd.DataFrame] = {}
        self._callbacks: list[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        """
        Daftarkan callback yang dipanggil saat candle baru ditutup
        dan indikator sudah dihitung.

        Args:
            callback: Fungsi async (pair, timeframe, dataframe) → None.
        """
        self._callbacks.append(callback)
        log.debug(f"Callback terdaftar: {callback.__name__}")

    def get_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame | None:
        """Ambil DataFrame untuk pair+timeframe tertentu."""
        return self.buffer.get((pair, timeframe))

    def initialize_buffer(self, pair: str, timeframe: str,
                          historical_candles: list[list]) -> None:
        """
        Inisialisasi buffer dengan data historis (dari REST API).

        Args:
            pair: Simbol pair, mis. "BTC/USDT".
            timeframe: Timeframe, mis. "15m".
            historical_candles: List of [timestamp, open, high, low, close, volume].
        """
        df = pd.DataFrame(
            historical_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Pastikan tipe data numerik
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Potong ke max_candles
        df = df.tail(self.max_candles).copy()

        # Hitung indikator
        df = self._compute_indicators(df)

        self.buffer[(pair, timeframe)] = df
        log.info(
            f"Buffer diinisialisasi — {pair} [{timeframe}] "
            f"dengan {len(df)} candle historis"
        )

    async def on_candle_close(self, pair: str, timeframe: str,
                              candle: dict) -> None:
        """
        Dipanggil saat candle baru ditutup dari WebSocket.

        Args:
            pair: Simbol pair.
            timeframe: Timeframe.
            candle: Dict dengan keys: timestamp, open, high, low, close, volume.
        """
        key = (pair, timeframe)

        # ── DEDUP: Skip jika candle dengan timestamp ini sudah ada ──
        ts = pd.to_datetime(candle["timestamp"], unit="ms")
        if key in self.buffer and ts in self.buffer[key].index:
            log.debug(f"Candle duplikat diabaikan — {pair} [{timeframe}] ts={ts}")
            return

        # Buat row baru
        new_row = pd.DataFrame([{
            "timestamp": ts,
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
        }])
        new_row.set_index("timestamp", inplace=True)

        # Append ke buffer atau buat baru
        if key in self.buffer:
            df = pd.concat([self.buffer[key], new_row])
        else:
            df = new_row

        # Potong ke max_candles
        df = df.tail(self.max_candles).copy()

        # Hitung ulang indikator
        df = self._compute_indicators(df)
        self.buffer[key] = df

        log.debug(
            f"Candle close — {pair} [{timeframe}] "
            f"O={candle['open']} H={candle['high']} "
            f"L={candle['low']} C={candle['close']}"
        )

        # Panggil semua callbacks
        for callback in self._callbacks:
            try:
                await callback(pair, timeframe, df)
            except Exception as e:
                log.error(f"Error di callback {callback.__name__}: {e}")

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hitung semua indikator teknikal pada DataFrame."""
        cfg = self.config.strategy

        # EMA fast & slow
        df = add_ema_columns(df, fast=cfg.ema_fast, slow=cfg.ema_slow)

        # CCI
        df = add_cci_column(df, length=cfg.cci_length)

        return df
