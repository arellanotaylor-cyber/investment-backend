from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Investment Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── WATCHLIST ────────────────────────────────────────────────────────────────

WATCHLIST = ["NVDA", "META", "GEV", "LLY", "AXON", "CEG", "ORCL", "XOM"]

# ─── SCORING ENGINE ───────────────────────────────────────────────────────────

def score_fundamental(info: dict) -> int:
    score = 50
    try:
        # P/E ratio (lower is better, but not negative)
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and 0 < pe < 15:   score += 15
        elif pe and 15 <= pe < 25: score += 8
        elif pe and 25 <= pe < 40: score += 3
        elif pe and pe >= 40:      score -= 5

        # Revenue growth
        rev_growth = info.get("revenueGrowth")
        if rev_growth:
            if rev_growth > 0.20:   score += 15
            elif rev_growth > 0.10: score += 8
            elif rev_growth > 0:    score += 3
            else:                   score -= 8

        # Profit margins
        margin = info.get("profitMargins")
        if margin:
            if margin > 0.25:   score += 12
            elif margin > 0.15: score += 6
            elif margin > 0.05: score += 2
            else:               score -= 5

        # Debt to equity
        dte = info.get("debtToEquity")
        if dte is not None:
            if dte < 50:    score += 8
            elif dte < 100: score += 3
            elif dte > 200: score -= 8

        # Return on equity
        roe = info.get("returnOnEquity")
        if roe:
            if roe > 0.20:   score += 10
            elif roe > 0.10: score += 5
            elif roe < 0:    score -= 10

    except Exception as e:
        logger.warning(f"Fundamental scoring error: {e}")

    return max(0, min(100, score))


def score_technical(hist) -> int:
    score = 50
    try:
        if hist is None or hist.empty or len(hist) < 20:
            return score

        closes = hist["Close"].tolist()
        current = closes[-1]

        # vs 20-day MA
        ma20 = sum(closes[-20:]) / 20
        if current > ma20 * 1.02:  score += 15
        elif current > ma20:       score += 8
        elif current < ma20 * 0.97: score -= 10

        # vs 50-day MA
        if len(closes) >= 50:
            ma50 = sum(closes[-50:]) / 50
            if current > ma50 * 1.03:  score += 12
            elif current > ma50:       score += 5
            elif current < ma50 * 0.95: score -= 8

        # Trend: higher highs over last 20 days
        first_half  = sum(closes[-20:-10]) / 10
        second_half = sum(closes[-10:]) / 10
        if second_half > first_half * 1.02:  score += 12
        elif second_half > first_half:       score += 5
        elif second_half < first_half * 0.97: score -= 8

        # Recent momentum (5-day)
        if len(closes) >= 5:
            mom = (closes[-1] - closes[-5]) / closes[-5]
            if mom > 0.05:    score += 10
            elif mom > 0.02:  score += 5
            elif mom < -0.05: score -= 10
            elif mom < -0.02: score -= 5

    except Exception as e:
        logger.warning(f"Technical scoring error: {e}")

    return max(0, min(100, score))


def compute_conviction(fundamental: int, technical: int, signal: int = 60) -> int:
    return round(fundamental * 0.35 + technical * 0.40 + signal * 0.25)


def get_sparkline(hist) -> list:
    if hist is None or hist.empty:
        return []
    closes = hist["Close"].dropna().tolist()
    # return last 8 closes rounded
    return [round(c, 2) for c in closes[-8:]]

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Investment Intelligence API online", "version": "1.0.0"}


@app.get("/quote/{ticker}")
def get_quote(ticker: str):
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        hist = t.history(period="5d", interval="1d")

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        closes = hist["Close"].tolist()
        current = round(closes[-1], 2)
        prev    = round(closes[-2], 2) if len(closes) >= 2 else current
        change  = round(((current - prev) / prev) * 100, 2) if prev else 0

        return {
            "ticker":      ticker.upper(),
            "name":        info.get("shortName", ticker),
            "price":       current,
            "prev_close":  prev,
            "change_pct":  change,
            "volume":      info.get("volume", 0),
            "avg_volume":  info.get("averageVolume", 0),
            "market_cap":  info.get("marketCap", 0),
            "sector":      info.get("sector", "Unknown"),
            "timestamp":   datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fundamentals/{ticker}")
def get_fundamentals(ticker: str):
    try:
        t    = yf.Ticker(ticker.upper())
        info = t.info

        return {
            "ticker":          ticker.upper(),
            "pe_trailing":     info.get("trailingPE"),
            "pe_forward":      info.get("forwardPE"),
            "eps":             info.get("trailingEps"),
            "revenue_growth":  info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin":   info.get("profitMargins"),
            "roe":             info.get("returnOnEquity"),
            "debt_to_equity":  info.get("debtToEquity"),
            "free_cashflow":   info.get("freeCashflow"),
            "52w_high":        info.get("fiftyTwoWeekHigh"),
            "52w_low":         info.get("fiftyTwoWeekLow"),
            "analyst_target":  info.get("targetMeanPrice"),
            "recommendation":  info.get("recommendationKey"),
            "fundamental_score": score_fundamental(info),
        }
    except Exception as e:
        logger.error(f"Fundamentals error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{ticker}")
def get_history(ticker: str, period: str = "3mo"):
    """
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, 10y
    """
    try:
        t    = yf.Ticker(ticker.upper())
        hist = t.history(period=period)

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No history for {ticker}")

        data = []
        for ts, row in hist.iterrows():
            data.append({
                "date":   ts.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]), 2),
                "high":   round(float(row["High"]), 2),
                "low":    round(float(row["Low"]), 2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        return {"ticker": ticker.upper(), "period": period, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/watchlist")
def get_watchlist():
    results = []
    for ticker in WATCHLIST:
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="3mo", interval="1d")

            closes  = hist["Close"].tolist() if not hist.empty else []
            current = round(closes[-1], 2) if closes else 0
            prev    = round(closes[-2], 2) if len(closes) >= 2 else current
            change  = round(((current - prev) / prev) * 100, 2) if prev else 0

            fund_score = score_fundamental(info)
            tech_score = score_technical(hist)
            sig_score  = 60  # placeholder until signal feeds are wired
            conviction = compute_conviction(fund_score, tech_score, sig_score)

            sparkline = get_sparkline(hist)

            results.append({
                "ticker":      ticker,
                "name":        info.get("shortName", ticker),
                "sector":      info.get("sector", "Unknown"),
                "price":       current,
                "change":      change,
                "score":       conviction,
                "fundamental": fund_score,
                "technical":   tech_score,
                "signal":      sig_score,
                "signals":     [],       # populated once signal feeds are live
                "sparkline":   sparkline,
                "market_cap":  info.get("marketCap", 0),
            })

        except Exception as e:
            logger.warning(f"Watchlist error for {ticker}: {e}")
            continue

    # sort by conviction score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"tickers": results, "updated": datetime.utcnow().isoformat()}


@app.get("/score/{ticker}")
def get_score(ticker: str):
    try:
        t    = yf.Ticker(ticker.upper())
        info = t.info
        hist = t.history(period="3mo", interval="1d")

        fund_score = score_fundamental(info)
        tech_score = score_technical(hist)
        sig_score  = 60
        conviction = compute_conviction(fund_score, tech_score, sig_score)

        return {
            "ticker":      ticker.upper(),
            "score":       conviction,
            "fundamental": fund_score,
            "technical":   tech_score,
            "signal":      sig_score,
            "breakdown": {
                "pe_score":      "positive" if (info.get("trailingPE") or 999) < 30 else "negative",
                "growth_score":  "positive" if (info.get("revenueGrowth") or 0) > 0.10 else "neutral",
                "margin_score":  "positive" if (info.get("profitMargins") or 0) > 0.15 else "neutral",
                "trend_score":   "positive" if tech_score > 65 else "neutral",
            }
        }
    except Exception as e:
        logger.error(f"Score error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
