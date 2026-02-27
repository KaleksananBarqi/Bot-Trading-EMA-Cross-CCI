"""
Unit test untuk modul strategy: EmaCciStrategy.
Memvalidasi logika sinyal BUY/SELL dan CCI filter.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.config.settings import BotConfig, StrategyConfig
from src.strategy.ema_cci_strategy import (
    EmaCciStrategy,
    SignalType,
    PositionState,
    TradeSignal,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def config() -> BotConfig:
    """Buat config default untuk testing."""
    return BotConfig(
        strategy=StrategyConfig(
            ema_fast=10,
            ema_slow=20,
            cci_length=20,
        )
    )


@pytest.fixture
def strategy(config: BotConfig) -> EmaCciStrategy:
    """Buat instance strategy."""
    return EmaCciStrategy(config)


def make_df(ema_10_prev: float, ema_10_curr: float,
            ema_20_prev: float, ema_20_curr: float,
            cci_curr: float, close_curr: float = 100.0) -> pd.DataFrame:
    """
    Buat DataFrame minimal dengan 2 baris untuk test crossover.

    Args:
        ema_10_prev/curr: Nilai EMA 10 pada candle sebelumnya dan saat ini.
        ema_20_prev/curr: Nilai EMA 20 pada candle sebelumnya dan saat ini.
        cci_curr: Nilai CCI saat ini.
        close_curr: Harga close saat ini.
    """
    timestamps = pd.date_range("2026-01-01", periods=2, freq="15min")
    df = pd.DataFrame({
        "open": [99.0, 99.5],
        "high": [101.0, 101.5],
        "low": [98.0, 98.5],
        "close": [100.0, close_curr],
        "volume": [1000.0, 1100.0],
        "ema_10": [ema_10_prev, ema_10_curr],
        "ema_20": [ema_20_prev, ema_20_curr],
        "cci_20": [0.0, cci_curr],  # CCI prev doesn't matter
    }, index=timestamps)
    return df


# ──────────────────────────────────────────────
# Test BUY Signal
# ──────────────────────────────────────────────

class TestBuySignal:
    """Test suite untuk sinyal BUY."""

    def test_buy_signal_ema_cross_up_cci_positive(self, strategy: EmaCciStrategy) -> None:
        """BUY: EMA10 cross above EMA20 + CCI > 0 → SINYAL BUY."""
        df = make_df(
            ema_10_prev=99.0, ema_10_curr=101.0,   # Cross up
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=50.0,                           # CCI > 0 ✅
            close_curr=101.5,
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.pair == "BTC/USDT"
        assert signal.timeframe == "15m"
        assert signal.entry_price == 101.5
        assert signal.cci_value == 50.0

    def test_buy_signal_rejected_cci_negative(self, strategy: EmaCciStrategy) -> None:
        """BUY: EMA10 cross above EMA20 + CCI < 0 → TIDAK ADA SINYAL."""
        df = make_df(
            ema_10_prev=99.0, ema_10_curr=101.0,   # Cross up
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=-30.0,                          # CCI < 0 ❌
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None

    def test_buy_signal_rejected_cci_zero(self, strategy: EmaCciStrategy) -> None:
        """BUY: EMA10 cross above EMA20 + CCI = 0 → TIDAK ADA SINYAL."""
        df = make_df(
            ema_10_prev=99.0, ema_10_curr=101.0,
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=0.0,                            # CCI = 0, bukan > 0
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None


# ──────────────────────────────────────────────
# Test SELL Signal
# ──────────────────────────────────────────────

class TestSellSignal:
    """Test suite untuk sinyal SELL."""

    def test_sell_signal_ema_cross_down_cci_negative(self, strategy: EmaCciStrategy) -> None:
        """SELL: EMA10 cross below EMA20 + CCI < 0 → SINYAL SELL."""
        df = make_df(
            ema_10_prev=101.0, ema_10_curr=99.0,    # Cross down
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=-40.0,                           # CCI < 0 ✅
            close_curr=98.5,
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)

        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.entry_price == 98.5
        assert signal.cci_value == -40.0

    def test_sell_signal_rejected_cci_positive(self, strategy: EmaCciStrategy) -> None:
        """SELL: EMA10 cross below EMA20 + CCI > 0 → TIDAK ADA SINYAL."""
        df = make_df(
            ema_10_prev=101.0, ema_10_curr=99.0,    # Cross down
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=20.0,                            # CCI > 0 ❌
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None


# ──────────────────────────────────────────────
# Test No Signal
# ──────────────────────────────────────────────

class TestNoSignal:
    """Test suite untuk skenario tanpa sinyal."""

    def test_no_cross_no_signal(self, strategy: EmaCciStrategy) -> None:
        """Tanpa crossover → tidak ada sinyal."""
        df = make_df(
            ema_10_prev=101.0, ema_10_curr=102.0,   # No cross (sudah di atas)
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=50.0,
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None

    def test_in_position_no_signal(self, strategy: EmaCciStrategy) -> None:
        """Jika sudah IN_POSITION → tidak generate sinyal baru."""
        strategy.set_state("BTC/USDT", "15m", PositionState.IN_POSITION)

        df = make_df(
            ema_10_prev=99.0, ema_10_curr=101.0,
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=50.0,
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None

    def test_nan_values_no_signal(self, strategy: EmaCciStrategy) -> None:
        """Data NaN → tidak ada sinyal."""
        df = make_df(
            ema_10_prev=float("nan"), ema_10_curr=101.0,
            ema_20_prev=100.0, ema_20_curr=100.5,
            cci_curr=50.0,
        )
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None

    def test_missing_columns_no_signal(self, strategy: EmaCciStrategy) -> None:
        """Kolom indikator hilang → tidak ada sinyal."""
        timestamps = pd.date_range("2026-01-01", periods=2, freq="15min")
        df = pd.DataFrame({
            "open": [99, 100],
            "high": [101, 102],
            "low": [98, 99],
            "close": [100, 101],
            "volume": [1000, 1100],
            # No indicator columns
        }, index=timestamps)
        signal = strategy.evaluate("BTC/USDT", "15m", df)
        assert signal is None


# ──────────────────────────────────────────────
# Test State Machine
# ──────────────────────────────────────────────

class TestStateMachine:
    """Test suite untuk state machine."""

    def test_initial_state_is_idle(self, strategy: EmaCciStrategy) -> None:
        """State awal harus IDLE."""
        state = strategy.get_state("BTC/USDT", "15m")
        assert state == PositionState.IDLE

    def test_set_and_get_state(self, strategy: EmaCciStrategy) -> None:
        """Set state harus bisa di-get kembali."""
        strategy.set_state("BTC/USDT", "15m", PositionState.IN_POSITION)
        assert strategy.get_state("BTC/USDT", "15m") == PositionState.IN_POSITION

    def test_different_pairs_independent_state(self, strategy: EmaCciStrategy) -> None:
        """State per pair harus independen."""
        strategy.set_state("BTC/USDT", "15m", PositionState.IN_POSITION)
        strategy.set_state("ETH/USDT", "15m", PositionState.IDLE)

        assert strategy.get_state("BTC/USDT", "15m") == PositionState.IN_POSITION
        assert strategy.get_state("ETH/USDT", "15m") == PositionState.IDLE
