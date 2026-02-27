"""
Binance WebSocket Feed — Real-time kline (candlestick) stream.
Menggunakan ccxt.pro (watchOHLCV) untuk menerima data candle secara async.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import ccxt.pro as ccxtpro
import pandas as pd

from src.config.settings import BotConfig
from src.utils.logger import log

if TYPE_CHECKING:
    from src.data.candle_manager import CandleManager


class WebSocketFeed:
    """
    Mengelola koneksi WebSocket ke Binance untuk streaming kline data.

    Menggunakan ccxt.pro watchOHLCV untuk setiap kombinasi pair × timeframe.
    Hanya memproses candle yang sudah ditutup (is_closed = True).

    Attributes:
        config: Konfigurasi bot.
        candle_manager: Instance CandleManager untuk menerima candle.
        exchange: Instance exchange ccxt.pro.
    """

    def __init__(self, config: BotConfig, candle_manager: "CandleManager") -> None:
        self.config = config
        self.candle_manager = candle_manager
        self.exchange: ccxtpro.Exchange | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def initialize(self) -> None:
        """
        Inisialisasi exchange ccxt.pro dan fetch data historis awal.
        """
        exchange_class = getattr(ccxtpro, self.config.exchange.name)

        exchange_params: dict = {
            "apiKey": self.config.binance_api_key,
            "secret": self.config.binance_api_secret,
            "options": {
                "defaultType": self.config.exchange.market_type,
            },
        }

        self.exchange = exchange_class(exchange_params)

        if self.config.exchange.testnet:
            self.exchange.enable_demo_trading(True)

        log.info(
            f"Exchange diinisialisasi — {self.config.exchange.name} "
            f"({'TESTNET' if self.config.exchange.testnet else 'PRODUCTION'}) "
            f"market_type={self.config.exchange.market_type}"
        )

        # Fetch data historis untuk warmup indikator
        await self._fetch_historical_data()

    async def _fetch_historical_data(self) -> None:
        """
        Fetch candle historis via REST API untuk mengisi buffer awal.
        Dibutuhkan agar indikator (EMA, CCI) punya data cukup saat pertama start.
        """
        assert self.exchange is not None

        for pair in self.config.pair_symbols:
            for tf in self.config.strategy.active_timeframes:
                try:
                    log.info(f"Fetching historical data — {pair} [{tf}]...")
                    candles = await self.exchange.fetch_ohlcv(
                        pair, tf, limit=100
                    )
                    self.candle_manager.initialize_buffer(pair, tf, candles)
                    log.info(
                        f"Historical data loaded — {pair} [{tf}] "
                        f"({len(candles)} candle)"
                    )
                except Exception as e:
                    log.error(
                        f"Gagal fetch historical data {pair} [{tf}]: {e}"
                    )

    async def start(self) -> None:
        """
        Mulai streaming WebSocket untuk semua pair × active_timeframes.
        Setiap kombinasi pair×tf dijalankan sebagai task asyncio terpisah.
        """
        self._running = True

        for pair in self.config.pair_symbols:
            for tf in self.config.strategy.active_timeframes:
                task = asyncio.create_task(
                    self._watch_kline(pair, tf),
                    name=f"ws_{pair}_{tf}"
                )
                self._tasks.append(task)
                log.info(f"WebSocket task dimulai — {pair} [{tf}]")

        log.info(
            f"WebSocket feed aktif — "
            f"{len(self.config.pair_symbols)} pairs × "
            f"{len(self.config.strategy.active_timeframes)} timeframes = "
            f"{len(self._tasks)} streams"
        )

        # Tunggu semua task (akan berjalan terus sampai stop)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _watch_kline(self, pair: str, timeframe: str) -> None:
        """
        Loop utama untuk menonton kline stream satu pair×timeframe.

        Args:
            pair: Simbol pair, mis. "BTC/USDT".
            timeframe: Timeframe, mis. "15m".
        """
        assert self.exchange is not None
        consecutive_errors = 0
        max_errors = 10

        while self._running:
            try:
                # watchOHLCV mengembalikan list of candles
                # Candle terakhir adalah yang sedang berjalan
                ohlcv_list = await self.exchange.watch_ohlcv(pair, timeframe)

                # Reset error counter saat sukses
                consecutive_errors = 0

                for candle_data in ohlcv_list:
                    # candle_data = [timestamp, open, high, low, close, volume]
                    timestamp, open_p, high, low, close, volume = candle_data

                    # ── DEDUP: Skip candle yang lebih tua/sama dari buffer terakhir ──
                    ts = pd.to_datetime(timestamp, unit="ms")
                    buf = self.candle_manager.get_dataframe(pair, timeframe)
                    if buf is not None and not buf.empty and ts <= buf.index[-1]:
                        continue

                    # ccxt.pro watchOHLCV memberikan candle yang sudah closed
                    # saat ada candle baru masuk (candle sebelumnya otomatis closed)
                    candle = {
                        "timestamp": timestamp,
                        "open": open_p,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }

                    await self.candle_manager.on_candle_close(
                        pair, timeframe, candle
                    )

            except ccxtpro.NetworkError as e:
                consecutive_errors += 1
                log.warning(
                    f"Network error pada {pair} [{timeframe}] "
                    f"(attempt {consecutive_errors}/{max_errors}): {e}"
                )
                if consecutive_errors >= max_errors:
                    log.error(
                        f"Terlalu banyak error berturut-turut pada "
                        f"{pair} [{timeframe}], menghentikan stream."
                    )
                    break
                await asyncio.sleep(min(consecutive_errors * 2, 30))

            except ccxtpro.ExchangeError as e:
                log.error(f"Exchange error pada {pair} [{timeframe}]: {e}")
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                log.info(f"Stream dibatalkan — {pair} [{timeframe}]")
                break

            except Exception as e:
                log.error(f"Unexpected error pada {pair} [{timeframe}]: {e}")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Hentikan semua WebSocket stream secara graceful."""
        self._running = False

        # Cancel semua task
        for task in self._tasks:
            task.cancel()

        # Tunggu semua task selesai
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Tutup koneksi exchange
        if self.exchange:
            await self.exchange.close()
            log.info("Exchange connection ditutup.")

        log.info("WebSocket feed dihentikan.")
