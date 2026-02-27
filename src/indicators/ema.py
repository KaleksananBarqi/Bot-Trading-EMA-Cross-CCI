"""
Kalkulasi Exponential Moving Average (EMA).
Mendukung EMA fast (10) dan EMA slow (20) dengan length configurable.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from src.utils.logger import log


def calculate_ema(df: pd.DataFrame, length: int, source: str = "close") -> pd.Series:
    """
    Hitung EMA untuk DataFrame OHLCV.

    Args:
        df: DataFrame dengan kolom OHLCV (minimal kolom `source`).
        length: Panjang periode EMA.
        source: Kolom sumber harga (default: "close").

    Returns:
        pd.Series berisi nilai EMA.

    Raises:
        ValueError: Jika kolom source tidak ada atau data tidak cukup.
    """
    if source not in df.columns:
        raise ValueError(f"Kolom '{source}' tidak ditemukan dalam DataFrame. "
                         f"Kolom tersedia: {list(df.columns)}")

    if len(df) < length:
        log.warning(
            f"Data tidak cukup untuk EMA({length}): "
            f"butuh {length} candle, tersedia {len(df)}"
        )
        return pd.Series([float("nan")] * len(df), index=df.index, name=f"ema_{length}")

    ema_series = ta.ema(df[source], length=length)
    ema_series.name = f"ema_{length}"
    return ema_series


def add_ema_columns(df: pd.DataFrame, fast: int = 10, slow: int = 20,
                    source: str = "close") -> pd.DataFrame:
    """
    Tambahkan kolom EMA fast dan EMA slow ke DataFrame.

    Args:
        df: DataFrame OHLCV.
        fast: Length EMA cepat (default: 10).
        slow: Length EMA lambat (default: 20).
        source: Kolom sumber harga.

    Returns:
        DataFrame yang sama dengan kolom ema_{fast} dan ema_{slow} ditambahkan.
    """
    df[f"ema_{fast}"] = calculate_ema(df, fast, source)
    df[f"ema_{slow}"] = calculate_ema(df, slow, source)

    log.debug(
        f"EMA({fast},{slow}) dihitung — "
        f"ema_{fast}={df[f'ema_{fast}'].iloc[-1]:.4f}, "
        f"ema_{slow}={df[f'ema_{slow}'].iloc[-1]:.4f}"
    )
    return df
