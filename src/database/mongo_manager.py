"""
MongoDB Manager — Async trade journal dan signal logging.
Menggunakan Motor (async driver untuk MongoDB).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import BotConfig
from src.utils.logger import log

if TYPE_CHECKING:
    from src.execution.position_tracker import Position
    from src.strategy.ema_cci_strategy import TradeSignal
    from src.risk.risk_manager import RiskParams


class MongoManager:
    """
    Mengelola koneksi MongoDB dan operasi CRUD untuk trade journal.

    Collections:
    - trades: Semua trade yang dieksekusi (entry, exit, PnL)
    - signals: Semua sinyal yang terdeteksi (termasuk yang ditolak CCI)

    Attributes:
        config: Konfigurasi bot.
        client: Motor async client.
        db: Database instance.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None

    async def connect(self) -> None:
        """Buat koneksi ke MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.config.mongo_uri)
            self.db = self.client[self.config.mongodb.database]

            # Test koneksi
            await self.client.admin.command("ping")
            log.info(
                f"MongoDB terhubung — "
                f"URI={self.config.mongo_uri.split('@')[-1]} "  # Sembunyikan credentials
                f"DB={self.config.mongodb.database}"
            )

            # Buat indexes
            await self._create_indexes()

        except Exception as e:
            log.error(f"Gagal terhubung ke MongoDB: {e}")
            raise

    async def _create_indexes(self) -> None:
        """Buat indexes untuk query yang efisien."""
        assert self.db is not None

        trades_col = self.db[self.config.mongodb.collection_trades]
        signals_col = self.db[self.config.mongodb.collection_signals]

        # Index untuk trades
        await trades_col.create_index([("pair", 1), ("entry_time", -1)])
        await trades_col.create_index([("timeframe", 1)])
        await trades_col.create_index([("exit_type", 1)])
        await trades_col.create_index([("entry_time", -1)])

        # Index untuk signals
        await signals_col.create_index([("pair", 1), ("timestamp", -1)])
        await signals_col.create_index([("signal_type", 1)])
        await signals_col.create_index([("accepted", 1)])

        log.debug("MongoDB indexes dibuat.")

    async def save_trade(self, position: "Position") -> str | None:
        """
        Simpan data trade ke MongoDB.

        Args:
            position: Objek Position yang baru dibuat atau sudah ditutup.

        Returns:
            ID dokumen yang disimpan, atau None jika gagal.
        """
        assert self.db is not None

        try:
            doc = {
                "pair": position.pair,
                "timeframe": position.timeframe,
                "side": position.side.value,
                "entry_price": position.entry_price,
                "sl_price": position.stop_loss,
                "tp_price": position.take_profit,
                "position_size": position.position_size,
                "entry_time": position.entry_time,
                "exit_price": position.exit_price,
                "exit_time": position.exit_time,
                "exit_type": position.exit_type,
                "pnl_pct": position.pnl,
                "cci_at_entry": position.cci_at_entry,
                "entry_mode": position.entry_mode,
                "order_id": position.order_id,
                "sl_order_id": position.sl_order_id,
                "tp_order_id": position.tp_order_id,
                "status": position.status.value,
                "updated_at": datetime.now(timezone.utc),
            }

            col = self.db[self.config.mongodb.collection_trades]
            result = await col.insert_one(doc)

            log.debug(f"Trade disimpan ke MongoDB — ID={result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            log.error(f"Gagal menyimpan trade ke MongoDB: {e}")
            return None

    async def update_trade_exit(self, pair: str, exit_price: float,
                                exit_type: str, pnl: float) -> None:
        """
        Update data trade saat posisi ditutup.

        Args:
            pair: Simbol pair.
            exit_price: Harga exit.
            exit_type: Tipe exit (tp, sl, manual).
            pnl: PnL dalam persen.
        """
        assert self.db is not None

        try:
            col = self.db[self.config.mongodb.collection_trades]
            await col.update_one(
                {"pair": pair, "status": "open"},
                {
                    "$set": {
                        "exit_price": exit_price,
                        "exit_time": datetime.now(timezone.utc),
                        "exit_type": exit_type,
                        "pnl_pct": pnl,
                        "status": "closed",
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                sort=[("entry_time", -1)],
            )
            log.debug(f"Trade di-update (closed) — {pair} exit={exit_type}")

        except Exception as e:
            log.error(f"Gagal update trade exit: {e}")

    async def save_signal(self, signal: "TradeSignal",
                          accepted: bool,
                          reject_reason: str = "") -> None:
        """
        Simpan sinyal ke MongoDB (termasuk yang ditolak).

        Args:
            signal: Sinyal trading.
            accepted: True jika sinyal dieksekusi, False jika ditolak.
            reject_reason: Alasan penolakan (jika accepted=False).
        """
        assert self.db is not None

        try:
            doc = {
                "pair": signal.pair,
                "timeframe": signal.timeframe,
                "signal_type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "ema_fast": signal.ema_fast_value,
                "ema_slow": signal.ema_slow_value,
                "cci": signal.cci_value,
                "accepted": accepted,
                "reject_reason": reject_reason,
                "timestamp": signal.timestamp or datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc),
            }

            col = self.db[self.config.mongodb.collection_signals]
            await col.insert_one(doc)

            status = "ACCEPTED" if accepted else f"REJECTED ({reject_reason})"
            log.debug(f"Signal disimpan — {signal.pair} {signal.signal_type.value} {status}")

        except Exception as e:
            log.error(f"Gagal menyimpan signal: {e}")

    async def get_trade_stats(self) -> dict[str, Any]:
        """
        Ambil statistik trading dari MongoDB.

        Returns:
            Dict dengan total trades, win rate, total PnL, dll.
        """
        assert self.db is not None

        col = self.db[self.config.mongodb.collection_trades]

        pipeline = [
            {"$match": {"status": "closed"}},
            {
                "$group": {
                    "_id": None,
                    "total_trades": {"$sum": 1},
                    "total_pnl": {"$sum": "$pnl_pct"},
                    "avg_pnl": {"$avg": "$pnl_pct"},
                    "wins": {
                        "$sum": {"$cond": [{"$gt": ["$pnl_pct", 0]}, 1, 0]}
                    },
                    "losses": {
                        "$sum": {"$cond": [{"$lte": ["$pnl_pct", 0]}, 1, 0]}
                    },
                }
            },
        ]

        result = await col.aggregate(pipeline).to_list(1)

        if not result:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
            }

        stats = result[0]
        total = stats["total_trades"]
        stats["win_rate"] = round(stats["wins"] / total * 100, 2) if total > 0 else 0
        stats.pop("_id", None)
        return stats

    async def disconnect(self) -> None:
        """Tutup koneksi MongoDB."""
        if self.client:
            self.client.close()
            log.info("MongoDB connection ditutup.")
