"""
Kalkulasi Commodity Channel Index (CCI).
Hanya garis utama CCI yang digunakan — SMA/Signal Line diabaikan.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from src.utils.logger import log


def calculate_cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Hitung CCI (Commodity Channel Index) untuk DataFrame OHLCV.

    CCI mengukur deviasi harga dari rata-rata statistiknya.
    Sistem ini hanya membaca garis utama CCI dan level 0.
    Signal Line / SMA TIDAK digunakan.

    Args:
        df: DataFrame dengan kolom high, low, close.
        length: Panjang periode CCI (default: 20).

    Returns:
        pd.Series berisi nilai CCI.

    Raises:
        ValueError: Jika kolom high/low/close tidak ada.
    """
    required_cols = {"high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    if len(df) < length:
        log.warning(
            f"Data tidak cukup untuk CCI({length}): "
            f"butuh {length} candle, tersedia {len(df)}"
        )
        return pd.Series([float("nan")] * len(df), index=df.index, name=f"cci_{length}")

    cci_series = ta.cci(df["high"], df["low"], df["close"], length=length)
    cci_series.name = f"cci_{length}"
    return cci_series


def add_cci_column(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """
    Tambahkan kolom CCI ke DataFrame.

    Args:
        df: DataFrame OHLCV.
        length: Panjang periode CCI.

    Returns:
        DataFrame yang sama dengan kolom cci_{length} ditambahkan.
    """
    df[f"cci_{length}"] = calculate_cci(df, length)

    log.debug(f"CCI({length}) dihitung — cci={df[f'cci_{length}'].iloc[-1]:.2f}")
    return df
