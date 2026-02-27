"""
Unit test untuk modul risk: RiskManager.
Memvalidasi kalkulasi SL, TP, position sizing, dan swing detection.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.config.settings import BotConfig, StrategyConfig, RiskConfig
from src.strategy.ema_cci_strategy import SignalType, TradeSignal
from src.risk.risk_manager import RiskManager, RiskParams


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def config_swing() -> BotConfig:
    """Config dengan SL mode swing_low."""
    return BotConfig(
        strategy=StrategyConfig(
            ema_fast=10,
            ema_slow=20,
            swing_lookback=10,
        ),
        risk=RiskConfig(
            risk_reward_ratio=2.0,
            sl_mode="swing_low",
            sl_buffer_pct=0.05,
            max_position_size_pct=5.0,
        ),
    )


@pytest.fixture
def config_ema() -> BotConfig:
    """Config dengan SL mode ema_slow."""
    return BotConfig(
        strategy=StrategyConfig(
            ema_fast=10,
            ema_slow=20,
            swing_lookback=10,
        ),
        risk=RiskConfig(
            risk_reward_ratio=2.0,
            sl_mode="ema_slow",
            sl_buffer_pct=0.05,
            max_position_size_pct=5.0,
        ),
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame OHLCV + indikator dengan swing points jelas."""
    timestamps = pd.date_range("2026-01-01", periods=15, freq="15min")
    # Data dengan swing low di index 5 (low=95) dan swing high di index 10 (high=108)
    df = pd.DataFrame({
        "open":  [100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 105, 107, 106, 108],
        "high":  [101, 100, 99, 98, 97, 96, 97, 98, 99, 100, 108, 106, 108, 107, 109],
        "low":   [99,  98,  97, 96, 95.5, 94.5, 95, 96, 97, 98, 99, 104, 106, 105, 107],
        "close": [100, 99, 98, 97, 96, 95.5, 96.5, 97.5, 98.5, 99.5, 107, 105.5, 107.5, 106, 108],
        "volume": [1000] * 15,
        "ema_10": [100] * 15,
        "ema_20": [99] * 15,
        "cci_20": [50] * 15,
    }, index=timestamps)
    return df


def make_buy_signal(entry_price: float = 108.0) -> TradeSignal:
    """Buat sinyal BUY untuk testing."""
    return TradeSignal(
        signal_type=SignalType.BUY,
        pair="BTC/USDT",
        timeframe="15m",
        entry_price=entry_price,
        ema_fast_value=101.0,
        ema_slow_value=99.0,
        cci_value=50.0,
    )


def make_sell_signal(entry_price: float = 95.0) -> TradeSignal:
    """Buat sinyal SELL untuk testing."""
    return TradeSignal(
        signal_type=SignalType.SELL,
        pair="BTC/USDT",
        timeframe="15m",
        entry_price=entry_price,
        ema_fast_value=98.0,
        ema_slow_value=100.0,
        cci_value=-40.0,
    )


# ──────────────────────────────────────────────
# Test SL Calculation
# ──────────────────────────────────────────────

class TestStopLoss:
    """Test suite untuk kalkulasi stop loss."""

    def test_buy_sl_swing_mode(self, config_swing: BotConfig,
                                sample_df: pd.DataFrame) -> None:
        """BUY SL dengan swing mode harus di bawah swing low."""
        rm = RiskManager(config_swing)
        signal = make_buy_signal(entry_price=108.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        assert result.stop_loss < signal.entry_price
        # SL harus di sekitar swing low area - buffer
        assert result.stop_loss > 0

    def test_sell_sl_swing_mode(self, config_swing: BotConfig,
                                 sample_df: pd.DataFrame) -> None:
        """SELL SL dengan swing mode harus di atas swing high."""
        rm = RiskManager(config_swing)
        signal = make_sell_signal(entry_price=95.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        assert result.stop_loss > signal.entry_price

    def test_buy_sl_ema_mode(self, config_ema: BotConfig,
                              sample_df: pd.DataFrame) -> None:
        """BUY SL dengan EMA mode harus di bawah EMA 20."""
        rm = RiskManager(config_ema)
        signal = make_buy_signal(entry_price=108.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        ema_20_value = sample_df["ema_20"].iloc[-1]
        assert result.stop_loss < ema_20_value


# ──────────────────────────────────────────────
# Test TP Calculation
# ──────────────────────────────────────────────

class TestTakeProfit:
    """Test suite untuk kalkulasi take profit."""

    def test_buy_tp_rr_ratio(self, config_swing: BotConfig,
                              sample_df: pd.DataFrame) -> None:
        """BUY TP harus sesuai R:R ratio 1:2."""
        rm = RiskManager(config_swing)
        signal = make_buy_signal(entry_price=108.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        assert result.take_profit > signal.entry_price

        # Verifikasi R:R ratio
        risk = signal.entry_price - result.stop_loss
        reward = result.take_profit - signal.entry_price
        ratio = reward / risk
        assert abs(ratio - 2.0) < 0.01  # R:R = 1:2

    def test_sell_tp_rr_ratio(self, config_swing: BotConfig,
                               sample_df: pd.DataFrame) -> None:
        """SELL TP harus sesuai R:R ratio 1:2."""
        rm = RiskManager(config_swing)
        signal = make_sell_signal(entry_price=95.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        assert result.take_profit < signal.entry_price

        # Verifikasi R:R ratio
        risk = result.stop_loss - signal.entry_price
        reward = signal.entry_price - result.take_profit
        ratio = reward / risk
        assert abs(ratio - 2.0) < 0.01


# ──────────────────────────────────────────────
# Test Position Sizing
# ──────────────────────────────────────────────

class TestPositionSizing:
    """Test suite untuk kalkulasi position sizing."""

    def test_position_size_proportional(self, config_swing: BotConfig,
                                        sample_df: pd.DataFrame) -> None:
        """Position size harus proporsional dengan balance."""
        rm = RiskManager(config_swing)
        signal = make_buy_signal(entry_price=108.0)

        result_small = rm.calculate(signal, sample_df, balance=1000)
        result_large = rm.calculate(signal, sample_df, balance=10000)

        assert result_small is not None
        assert result_large is not None
        # Size dengan balance besar harus lebih besar
        assert result_large.position_size > result_small.position_size

    def test_position_size_positive(self, config_swing: BotConfig,
                                     sample_df: pd.DataFrame) -> None:
        """Position size harus selalu positif."""
        rm = RiskManager(config_swing)
        signal = make_buy_signal(entry_price=108.0)
        result = rm.calculate(signal, sample_df, balance=10000)

        assert result is not None
        assert result.position_size > 0

    def test_zero_balance(self, config_swing: BotConfig,
                           sample_df: pd.DataFrame) -> None:
        """Balance 0 → position size 0 (tapi masih valid)."""
        rm = RiskManager(config_swing)
        signal = make_buy_signal(entry_price=108.0)
        result = rm.calculate(signal, sample_df, balance=0)

        assert result is not None
        assert result.position_size == 0


# ──────────────────────────────────────────────
# Test Swing Detection
# ──────────────────────────────────────────────

class TestSwingDetection:
    """Test suite untuk deteksi swing high/low."""

    def test_find_swing_low(self) -> None:
        """Harus menemukan local minimum."""
        from src.risk.risk_manager import RiskManager

        timestamps = pd.date_range("2026-01-01", periods=5, freq="15min")
        df = pd.DataFrame({
            "low": [100, 98, 95, 97, 99],  # Swing low di index 2
            "high": [105, 103, 100, 102, 104],
        }, index=timestamps)

        swing_low = RiskManager._find_swing_low(df)
        assert swing_low is not None
        assert swing_low == 95.0

    def test_find_swing_high(self) -> None:
        """Harus menemukan local maximum."""
        from src.risk.risk_manager import RiskManager

        timestamps = pd.date_range("2026-01-01", periods=5, freq="15min")
        df = pd.DataFrame({
            "low": [95, 97, 99, 97, 95],
            "high": [100, 103, 108, 105, 102],  # Swing high di index 2
        }, index=timestamps)

        swing_high = RiskManager._find_swing_high(df)
        assert swing_high is not None
        assert swing_high == 108.0

    def test_no_swing_in_monotonic_data(self) -> None:
        """Data monoton → tidak ada swing point."""
        from src.risk.risk_manager import RiskManager

        timestamps = pd.date_range("2026-01-01", periods=5, freq="15min")
        df = pd.DataFrame({
            "low": [100, 101, 102, 103, 104],   # Monoton naik
            "high": [105, 106, 107, 108, 109],
        }, index=timestamps)

        assert RiskManager._find_swing_low(df) is None
        assert RiskManager._find_swing_high(df) is None
