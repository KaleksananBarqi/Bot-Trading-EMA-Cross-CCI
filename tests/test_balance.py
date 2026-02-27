"""
Unit test untuk pengecekan balance di modul utama (main.py > TradingBot).
Tujuan: Memverifikasi parsing response CCXT saat fetch_balance().
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.config.settings import BotConfig
from src.main import TradingBot


@pytest.fixture
def mock_config() -> BotConfig:
    """Fixture untuk konfirgurasi dasar."""
    return BotConfig()


@pytest.fixture
def bot(mock_config: BotConfig) -> TradingBot:
    """Fixture untuk instance TradingBot."""
    # Matikan komponen tambahan agar inisialisasi lebih cepat & terisolasi
    bot_instance = TradingBot(mock_config)
    
    # Mocking obyek ws_feed dan exchange untuk menyimulasikan state _get_balance
    bot_instance.ws_feed = MagicMock()
    bot_instance.ws_feed.exchange = AsyncMock()
    return bot_instance


@pytest.mark.asyncio
class TestBalanceCheck:
    """Test suite untuk verifikasi fungsionalitas `_get_balance`."""

    async def test_get_balance_success(self, bot: TradingBot) -> None:
        """Skenario sukses saat fetch_balance mengembalikan data USDT dengan normal."""
        # Simulasi return dari Binance Futures melalui CCXT
        bot.ws_feed.exchange.fetch_balance.return_value = {
            "info": {},
            "free": {
                "USDT": 1500.5,
                "BUSD": 100.0,
            },
            "used": {
                "USDT": 50.0
            },
            "total": {
                "USDT": 1550.5
            }
        }
        
        balance = await bot._get_balance()
        assert balance == 1500.5

    async def test_get_balance_no_usdt(self, bot: TradingBot) -> None:
        """Skenario sukses fetch exchange, tapi tidak memiliki saldo USDT sama sekali."""
        # Dictionary 'free' tanpa 'USDT'
        bot.ws_feed.exchange.fetch_balance.return_value = {
            "free": {
                "BTC": 1.5,
            }
        }
        
        balance = await bot._get_balance()
        assert balance == 0.0
        
    async def test_get_balance_empty_response(self, bot: TradingBot) -> None:
        """Skenario ketika exchange mengembalikan dict kosong."""
        bot.ws_feed.exchange.fetch_balance.return_value = {}
        
        balance = await bot._get_balance()
        assert balance == 0.0

    async def test_get_balance_api_error(self, bot: TradingBot) -> None:
        """Skenario ketika CCXT fetch_balance melempar exception (misal: network/rate limit)."""
        import ccxt
        bot.ws_feed.exchange.fetch_balance.side_effect = ccxt.NetworkError("Connection reset by peer")
        
        balance = await bot._get_balance()
        # Harus memfallback exception dan nilai menjadi 0.0 secara gracefully
        assert balance == 0.0
