# Bot Trading EMA Cross + CCI

Bot trading otomatis untuk **Binance Futures** (USDⓈ-M) yang menggunakan strategi **EMA 10/20 Crossover** dengan filter momentum **CCI(20)**.

## Arsitektur & Teknologi

* **Bahasa**: Python 3.11+ (Asyncio architecture)
* **Exchange API**: `ccxt.pro` (WebSocket) & `ccxt` (REST)
* **Indikator**: `pandas-ta` (EMA, CCI)
* **Manajemen Data**: `pandas` DataFrame dengan local rolling buffer
* **Database**: MongoDB (`motor` async driver) untuk trade journal
* **Notifikasi**: Telegram (`python-telegram-bot`)

## Logika Strategi Inti

### ENTRY (Buy / Long)
1. **Trigger**: EMA 10 (Fast) memotong EMA 20 (Slow) ke arah **ATAS** (Cross Up).
2. **Filter (Hard Gate)**: Pada saat yang bersamaan, garis utama CCI(20) **wajib** berada di **ATAS 0**. Jika CCI < 0, sinyal diabaikan.

### ENTRY (Sell / Short)
1. **Trigger**: EMA 10 (Fast) memotong EMA 20 (Slow) ke arah **BAWAH** (Cross Down).
2. **Filter (Hard Gate)**: Pada saat yang bersamaan, garis utama CCI(20) **wajib** berada di **BAWAH 0**. Jika CCI > 0, sinyal diabaikan.

### MANAJEMEN RISIKO (SL & TP)
1. **Risk to Reward (R:R)**: Dikunci pada **1:2** (Target profit 2x lebih besar dari jarak Stop Loss).
2. **Stop Loss Mode**: Mendukung 2 mode (diatur di `config.yaml`):
   * `swing_low`: SL diletakkan di bawah *swing low* (untuk Buy) atau di atas *swing high* (untuk Sell) terdekat dalam *N* candle terakhir (default N=10).
   * `ema_slow`: SL diletakkan sedikit di bawah area EMA 20 (untuk Buy) atau di atas EMA 20 (untuk Sell).
3. **Position Sizing**: Dinamis berdasarkan % risiko modal per trade (default 5% dari saldo).

## Mode Eksekusi (Entry Mode)

Ditentukan pada `config.yaml` > `strategy.entry_mode`:

1. **`close` (Default):** Market order langsung dieksekusi saat candle yang memicu sinyal ditutup (close).
2. **`pullback`:** Limit order diletakkan di **harga EMA 10**. Bot akan menunggu harga terkoreksi (pullback) menyentuh limit ini. Jika dalam beberapa candle tidak tersentuh, order otomatis dibatalkan.

## Struktur Direktori

```
├── config.yaml               # Konfigurasi parameter (Strategy, Risk, TF)
├── .env                      # Secrets (API Keys, Telegram Token, Mongo URI)
├── src/
│   ├── main.py               # Orchestrator utama bot
│   ├── config/               # Settings validator (Pydantic v2)
│   ├── data/                 # WebSocket Feed & Candle Buffer Manager
│   ├── indicators/           # Modul kalkulasi EMA dan CCI (pandas-ta)
│   ├── strategy/             # Logika core EMA Cross & CCI Filter
│   ├── execution/            # Order Management (ccxt) & Position Tracker
│   ├── risk/                 # Risiko, Stop Loss, TP, Position Sizing
│   ├── database/             # MongoDB Journaling
│   ├── notifications/        # Telegram Notifier async
│   └── utils/                # Loguru logger terpusat
└── tests/                    # Unit tests (Pytest)
```

## Persiapan & Instalasi

### 1. Requirements Eksternal
* Python 3.10 atau lebih baru
* MongoDB Server (berjalan lokal atau Atlas)

### 2. Instalasi Environment
```bash
# Clone repository
git clone https://github.com/KaleksananBarqi/Bot-Trading-EMA-Cross-CCI.git
cd Bot-Trading-EMA-Cross-CCI

# Buat virtual environment
python -m venv .venv

# Aktivasi (Windows)
.venv\Scripts\activate
# Aktivasi (Mac/Linux)
# source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Konfigurasi
1. Buka file `.env`.
2. Isi rahasia berikut:
   ```env
   BINANCE_API_KEY=kunci_api_anda
   BINANCE_API_SECRET=rahasia_api_anda
   TELEGRAM_TOKEN=token_bot_dari_botfather
   TELEGRAM_CHAT_ID=id_telegram_anda
   TELEGRAM_MESSAGE_THREAD_ID=id_topik_grup_opsional
   MONGO_URI=mongodb://localhost:27017
   ```
3. Buka `config.yaml` untuk menyesuaikan parameter strategi. Fitur penting:
   * `testnet: true` (Sangat disarankan memakai Testnet di awal)
   * `active_timeframes`: Tentukan TF mana bot harus aktif (misalnya `["15m", "1h"]`). TF `1m` diblokir oleh sistem untuk menghindari *noise*.
   * `pairs`: Terapkan pada pair yang dituju (`symbol`), atur `leverage` (misal 20) dan `margin_mode` (isolated/crossed).

## Menjalankan Bot

Untuk menjalankan bot utama:
```bash
ema-cci-bot
# atau
python -m src.main
```

Bot akan melakukan *warmup* (mengambil data historis) agar indikator langsung terhitung, lalu otomatis mendengarkan stream WebSocket Binance.

## Menjalankan Unit Tests

Untuk memverifikasi semua fungsi berjalan dengan benar:
```bash
pytest tests/ -v
```
Ini akan mengeksekusi puluhan skenario tes otomatis meliputi sinyal strategy, crossover, validasi filter CCI, hingga perhitungan Stop Loss dan Position Sizing.
