#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
import nest_asyncio

# ================= CONFIG =================
TELEGRAM_TOKEN = "8261369309:AAEKleXLJfo6Xo1ym5wW8f6yf_kCSuhLvGo"
CHAT_ID = "940078832"

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

BYBIT_SYMBOLS_URL = "https://api.bybit.com/v5/market/instruments-info"
BYBIT_KLINES_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"

IMPULSE_THRESHOLD = 15.0  # відсоток зміни для сигналу
TF_MAP = {"15m": 15, "1h": 60, "4h": 240}
WORKING_TFS = set(TF_MAP.keys())

LOOKBACK = 3
EMA_PERIOD = 5
POLL_INTERVAL = 30
COOLDOWN_SECONDS = 1800
MAX_CONCURRENT = 12

# ================= UTIL =================
def now_str():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now_str()}] {msg}")

async def safe_get(session, url, params=None, tries=3, timeout=12):
    for attempt in range(tries):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = await resp.text()
                return resp.status, data
        except Exception as e:
            log(f"safe_get error ({attempt+1}/{tries}) {url}: {e}")
            if attempt + 1 < tries:
                await asyncio.sleep(0.6)
            else:
                return None, None
    return None, None

# ================= SYMBOLS =================
async def load_symbols(session):
    status, data = await safe_get(session, BYBIT_SYMBOLS_URL, {"category": "linear"})
    if status == 200 and isinstance(data, dict):
        try:
            return [
                r["symbol"]
                for r in data.get("result", {}).get("list", [])
                if r.get("quoteCoin") == "USDT" and r.get("status") == "Trading"
            ]
        except Exception:
            return []
    return []

def minutes_to_interval(m):
    return {15: "15", 60: "60", 240: "240"}.get(m, str(m))

async def fetch_klines(session, symbol, minutes, limit=LOOKBACK+EMA_PERIOD+5):
    status, data = await safe_get(
        session,
        BYBIT_KLINES_URL,
        {
            "category": "linear",
            "symbol": symbol,
            "interval": minutes_to_interval(minutes),
            "limit": str(limit),
        },
    )
    if status == 200 and isinstance(data, dict):
        rows = data.get("result", {}).get("list", [])
        if rows and len(rows) >= 2:
            df = pd.DataFrame(rows).iloc[:, :6]
            df.columns = ["start", "open", "high", "low", "close", "volume"]
            df[["close", "open", "high", "low"]] = df[["close", "open", "high", "low"]].apply(
                pd.to_numeric, errors="coerce"
            )
            return df
    return None

# ================= TICKER =================
async def fetch_mark_price(session, symbol):
    status, data = await safe_get(
        session,
        BYBIT_TICKERS_URL,
        {"category": "linear", "symbol": symbol},
    )
    if status == 200 and isinstance(data, dict):
        try:
            lst = data.get("result", {}).get("list", [])
            if lst:
                return float(lst[0].get("markPrice") or lst[0].get("lastPrice"))
        except Exception:
            return None
    return None

# ================= ANALYZE =================
def pick_best_tf(multi):
    best_tf, best_score, best_df = None, 0, None
    for tf, df in multi.items():
        if tf not in WORKING_TFS:
            continue
        if df is None or len(df) < 2:
            continue
        prev_close, last_close = df.iloc[-2]["close"], df.iloc[-1]["close"]
        imp = (last_close - prev_close) / prev_close * 100
        if abs(imp) >= IMPULSE_THRESHOLD and abs(imp) > best_score:
            best_tf, best_score, best_df = tf, imp, df
    if best_tf:
        return best_tf, best_score, best_df
    return None, None, None

def estimate_end(df, impulse_pct, interval):
    if df is None or len(df) < 2:
        return None
    df = df.copy()
    df["pct_change"] = df["close"].pct_change() * 100
    avg = df["pct_change"].iloc[-LOOKBACK:].abs().mean()
    if avg == 0:
        return None
    remaining_minutes = abs(impulse_pct) / avg * interval
    return datetime.now() + timedelta(minutes=max(remaining_minutes, interval))

def compute_ema(series, period=EMA_PERIOD):
    return series.ewm(span=period, adjust=False).mean()

# ================= AI =================
async def get_ai_analysis(symbol, tf, entry_price, impulse):
    prompt = (
        f"Оціни {symbol} {tf}. Вхід: {entry_price:.8f}, імпульс: {impulse:.2f}%. "
        f"Дай коротко поради трейдингу українською, 1-2 речення, без пояснення свічок."
    )
    async with aiohttp.ClientSession() as session:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with session.post(GROQ_CHAT_URL, headers=headers, json=payload) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

# ================= BUILD MESSAGE =================
def build_message(symbol, tf, entry_price, impulse, end_time, ai_text=""):
    adj = abs(impulse)/100
    tp = entry_price * (1 + adj*1.5 if impulse > 0 else 1 - adj*1.5)
    sl = entry_price * (1 - adj*0.5 if impulse > 0 else 1 + adj*0.5)
    risk = round(adj*50,1)
    potential = round(adj*100*1.5,1)
    direction = "🟢 ЛОНГ" if impulse > 0 else "🔻 ШОРТ"

    lines = [  
        f"🔥 <b>{symbol} {tf}</b>",  
        f"➡ Вхід: {entry_price:.8f}",  
        f"⚡ Імпульс: {impulse:.2f}%",  
        f"{direction} | 🎯 TP: {tp:.8f} | ⚠ SL: {sl:.8f} | Ризик/Потенціал: {risk}%/{potential}%",  
    ]  
    if ai_text:  
        lines.append(f"🤖 AI: {ai_text}")  
    if end_time:  
        lines.append(f"⏳ Кінець імпульсу: {end_time.strftime('%Y-%m-%d %H:%M')}")  
    lines.append(f"⏱ Час: {now_str()}")  
    return "\n".join(lines)

# ================= TELEGRAM =================
async def send_telegram(session, text):
    try:
        await session.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
        )
        return True
    except Exception as e:
        log(f"Telegram error: {e}")
        return False

# ================= WORKFLOW =================
async def handle_symbol(session, symbol, last_sent):
    try:
        tasks = [fetch_klines(session, symbol, m) for m in TF_MAP.values()]
        results = await asyncio.gather(*tasks)
        multi = {tf: r for tf, r in zip(TF_MAP.keys(), results)}

        tf, impulse, df = pick_best_tf(multi)
        if not tf:
            return False

        mark_price = await fetch_mark_price(session, symbol)
        if mark_price is None:
            return False

        # --- Фільтр розвороту тренду ---
        recent = df["close"].iloc[-LOOKBACK:]
        trend = recent.diff().sum()
        if impulse > 0 and trend <= 0:
            return False
        if impulse < 0 and trend >= 0:
            return False

        # --- Точка входу з EMA ---
        ema_series = compute_ema(df["close"])
        ema = ema_series.iloc[-LOOKBACK:].mean()
        if impulse > 0:
            pivot = df["low"].iloc[-LOOKBACK:].min()
            entry_price = max(pivot, mark_price, ema)
        else:
            pivot = df["high"].iloc[-LOOKBACK:].max()
            entry_price = min(pivot, mark_price, ema)

        now_time = time.time()
        if now_time < last_sent.get(symbol, 0) + COOLDOWN_SECONDS:
            return False

        end_time = estimate_end(df, impulse, TF_MAP[tf])
        ai_text = await get_ai_analysis(symbol, tf, entry_price, impulse)
        text = build_message(symbol, tf, entry_price, impulse, end_time, ai_text)

        if await send_telegram(session, text):
            last_sent[symbol] = now_time
            log(f"Сигнал відправлено {symbol} {tf} {impulse:.2f}%")
            return True

    except Exception as e:
        log(f"Помилка {symbol}: {e}")
    return False

# ================= MAIN =================
async def main():
    log("Старт моніторингу Bybit (15m, 1h, 4h)")
    async with aiohttp.ClientSession() as session:
        symbols = await load_symbols(session)
        if not symbols:
            log("Символи не завантажено")
            return
        last_sent = {}
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def worker(sym):
            async with sem:
                return await handle_symbol(session, sym, last_sent)

        while True:
            start = time.time()
            tasks = [asyncio.create_task(worker(s)) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            sent = sum(1 for r in results if r is True)
            log(f"Скан завершено. Сигнали: {sent}. Час: {time.time() - start:.1f}s")
            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Зупинено користувачем")
