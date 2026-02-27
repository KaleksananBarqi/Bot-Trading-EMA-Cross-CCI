"""
Loader dan validasi konfigurasi menggunakan Pydantic v2.
Memuat config.yaml + .env, lalu memvalidasi semua parameter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import log

# ──────────────────────────────────────────────
# Sub-model konfigurasi
# ──────────────────────────────────────────────

class ExchangeConfig(BaseModel):
    """Konfigurasi exchange (Binance)."""
    name: str = "binance"
    market_type: Literal["future", "spot"] = "future"
    testnet: bool = True


class StrategyConfig(BaseModel):
    """Parameter strategi EMA Cross + CCI."""
    ema_fast: int = Field(default=10, ge=1, description="Length EMA cepat")
    ema_slow: int = Field(default=20, ge=1, description="Length EMA lambat")
    cci_length: int = Field(default=20, ge=1, description="Length CCI")
    valid_timeframes: list[str] = Field(default=["15m", "1h", "4h"])
    active_timeframes: list[str] = Field(default=["15m"])
    entry_mode: Literal["close", "pullback"] = "close"
    swing_lookback: int = Field(default=10, ge=3, description="Candle lookback untuk swing H/L")
    pullback_ttl_candles: int = Field(default=3, ge=1, description="TTL pullback order dalam jumlah candle")

    @field_validator("ema_fast")
    @classmethod
    def ema_fast_less_than_slow(cls, v: int, info) -> int:
        # Validasi dilakukan di model_validator karena butuh akses ke ema_slow
        return v

    @field_validator("active_timeframes")
    @classmethod
    def validate_active_timeframes(cls, v: list[str], info) -> list[str]:
        """Pastikan tidak ada 1m dan semua TF valid format."""
        forbidden = {"1m", "1min"}
        for tf in v:
            if tf.lower() in forbidden:
                raise ValueError(
                    f"Timeframe '{tf}' DILARANG! Hanya 15m, 1h, 4h yang diperbolehkan."
                )
        return v

    @model_validator(mode="after")
    def validate_strategy_consistency(self) -> "StrategyConfig":
        """Validasi konsistensi parameter strategi."""
        # EMA fast harus lebih kecil dari slow
        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"ema_fast ({self.ema_fast}) harus lebih kecil dari "
                f"ema_slow ({self.ema_slow})"
            )
        # active_timeframes harus subset dari valid_timeframes
        valid_set = set(self.valid_timeframes)
        for tf in self.active_timeframes:
            if tf not in valid_set:
                raise ValueError(
                    f"active_timeframe '{tf}' tidak ada dalam "
                    f"valid_timeframes {self.valid_timeframes}"
                )
        return self


class RiskConfig(BaseModel):
    """Parameter manajemen risiko."""
    risk_reward_ratio: float = Field(default=2.0, gt=0)
    sl_mode: Literal["swing_low", "ema_slow"] = "swing_low"
    sl_buffer_pct: float = Field(default=0.05, ge=0, description="Buffer SL dalam %")
    max_position_size_pct: float = Field(default=5.0, gt=0, le=100)


class TelegramConfig(BaseModel):
    """Konfigurasi notifikasi Telegram."""
    enabled: bool = True
    notify_signals: bool = True
    notify_fills: bool = True
    notify_sl_tp: bool = True
    notify_errors: bool = True


class MongoDBConfig(BaseModel):
    """Konfigurasi MongoDB."""
    database: str = "ema_cci_bot"
    collection_trades: str = "trades"
    collection_signals: str = "signals"


class LoggingConfig(BaseModel):
    """Konfigurasi logging."""
    level: str = "INFO"
    file: str | None = "logs/bot.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


# ──────────────────────────────────────────────
# Model utama
# ──────────────────────────────────────────────

class BotConfig(BaseModel):
    """Konfigurasi utama bot — menggabungkan semua sub-konfigurasi."""
    exchange: ExchangeConfig = ExchangeConfig()
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()
    pairs: list[str] = Field(default=["BTC/USDT"])
    telegram: TelegramConfig = TelegramConfig()
    mongodb: MongoDBConfig = MongoDBConfig()
    logging: LoggingConfig = LoggingConfig()

    # Secrets dari .env (akan diisi setelah load)
    binance_api_key: str = ""
    binance_api_secret: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    mongo_uri: str = "mongodb://localhost:27017"


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> BotConfig:
    """
    Memuat konfigurasi dari file YAML + .env.

    Args:
        config_path: Path ke config.yaml.
        env_path: Path ke file .env.

    Returns:
        BotConfig yang sudah tervalidasi.

    Raises:
        FileNotFoundError: Jika config.yaml tidak ditemukan.
        ValidationError: Jika konfigurasi tidak valid.
    """
    import os

    # Load .env
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)
        log.info(f"Loaded .env dari: {env_file.resolve()}")
    else:
        log.warning(f"File .env tidak ditemukan: {env_file.resolve()}")

    # Load config.yaml
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    log.info(f"Loaded config.yaml dari: {cfg_file.resolve()}")

    # Inject secrets dari environment variables
    raw_config["binance_api_key"] = os.getenv("BINANCE_API_KEY", "")
    raw_config["binance_api_secret"] = os.getenv("BINANCE_API_SECRET", "")
    raw_config["telegram_bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    raw_config["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")
    raw_config["mongo_uri"] = os.getenv("MONGO_URI", "mongodb://localhost:27017")

    if "mongodb" not in raw_config:
        raw_config["mongodb"] = {}
    if os.getenv("MONGO_DB_NAME"):
        raw_config["mongodb"]["database"] = os.getenv("MONGO_DB_NAME")
    if os.getenv("MONGO_COLLECTION_NAME"):
        raw_config["mongodb"]["collection_trades"] = os.getenv("MONGO_COLLECTION_NAME")

    # Parse & validasi
    config = BotConfig(**raw_config)

    log.info(
        f"Config tervalidasi — "
        f"exchange={config.exchange.name} "
        f"testnet={config.exchange.testnet} "
        f"pairs={config.pairs} "
        f"active_tf={config.strategy.active_timeframes} "
        f"entry_mode={config.strategy.entry_mode}"
    )

    return config
