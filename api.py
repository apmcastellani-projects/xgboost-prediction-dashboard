# api.py
# ---------------------------------------------------------------------------
# FastAPI backend exposing prediction, backtest, and history endpoints.
# Run with:  uvicorn api:app --reload --port 8000
# ---------------------------------------------------------------------------

from __future__ import annotations
import time
import logging
from functools import lru_cache

import os
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple

from data_adapter import get_adapter, OHLCV
from ml_engine   import predict, backtest, PredictionResult, BacktestResult, optimize_portfolio, OptimizationResult, calculate_custom_portfolio, CustomPortfolioResult

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Stock Prediction API",
    description = "Multi-horizon XGBoost price-return predictions with realistic backtest.",
    version     = "1.0.0",
)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("FRONTEND_URL", "*").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ALLOWED_ORIGINS,
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


from concurrent.futures import ThreadPoolExecutor

class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.cache = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self._lock:
            self.cache[key] = (value, time.time())


# Caches with 1 hour TTL
ohlcv_cache = SimpleTTLCache(ttl_seconds=3600)
predict_cache = SimpleTTLCache(ttl_seconds=3600)
backtest_cache = SimpleTTLCache(ttl_seconds=3600)


def _cached_ohlcv(ticker: str, years: int = 5) -> OHLCV:
    """Cache raw market data for 1 hour."""
    key = (ticker.upper(), years)
    cached = ohlcv_cache.get(key)
    if cached is not None:
        return cached

    adapter = get_adapter("yfinance")
    log.info(f"Fetching {ticker} via {adapter.name()} ({years}y)")
    res = adapter.fetch(ticker.upper(), years=years)
    ohlcv_cache.set(key, res)
    return res


def _cached_predict(ticker: str, horizon: str, years: int) -> PredictionResult:
    """Cache prediction for 1 hour."""
    key = (ticker.upper(), horizon.lower(), years)
    cached = predict_cache.get(key)
    if cached is not None:
        return cached
    log.info(f"Training prediction model: {ticker} / {horizon} / {years}y")
    ohlcv = _cached_ohlcv(ticker, years)
    res = predict(ohlcv, horizon)
    predict_cache.set(key, res)
    return res


def _cached_backtest(ticker: str, horizon: str, years: int) -> BacktestResult:
    """Cache backtest results for 1 hour."""
    key = (ticker.upper(), horizon.lower(), years)
    cached = backtest_cache.get(key)
    if cached is not None:
        return cached
    log.info(f"Running backtest: {ticker} / {horizon} / {years}y")
    ohlcv = _cached_ohlcv(ticker, years)
    res = backtest(ohlcv, horizon)
    backtest_cache.set(key, res)
    return res


optimize_cache = SimpleTTLCache(ttl_seconds=3600)

def _cached_optimize(tickers_str: str, months: int, years: int, min_w: float = 0.0, max_w: float = 1.0) -> OptimizationResult:
    """Cache portfolio optimization results for 1 hour."""
    key = (tickers_str.upper(), months, years, min_w, max_w)
    cached = optimize_cache.get(key)
    if cached is not None:
        return cached
        
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
        
    def _fetch_and_train(t: str):
        ohlcv = _cached_ohlcv(t, years)
        res = _cached_predict(t, "monthly", years)
        return t, ohlcv, res.pred_log_return, res.market_regime

    ohlcv_dict = {}
    predictions = {}
    regimes = {}
    with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as executor:
        for t, ohlcv, pred_lr, regime in executor.map(_fetch_and_train, tickers):
            ohlcv_dict[t] = ohlcv
            predictions[t] = pred_lr
            regimes[t] = regime
        
    res = optimize_portfolio(ohlcv_dict, predictions, months, min_w, max_w)
    res.asset_regimes = regimes
    optimize_cache.set(key, res)
    return res


# ── Response schemas ──────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    title:     str
    publisher: str
    link:      str
    date:      str
    sentiment: float


class PredictionResponse(BaseModel):
    ticker:           str
    horizon:          str
    last_price:       float
    pred_price:       float
    pred_return_pct:  float
    ci_lower:         float
    ci_upper:         float
    hist_dates:       List[str]
    hist_prices:      List[float]
    wf_dates:         List[str]
    wf_pred_prices:   List[float]
    wf_actual_prices: List[float]
    rmse:             float
    vol_21_pct:       float
    elapsed_s:        float
    market_regime:    str
    shap_values:      Dict[str, float]
    news:             List[NewsItem]
    mc_paths:         List[List[float]]


class BacktestResponse(BaseModel):
    ticker:           str
    horizon:          str
    total_return_pct: float
    buy_hold_pct:     float
    sharpe_ratio:     float
    max_drawdown_pct: float
    n_trades:         int
    dates:            List[str]
    equity_curve:     List[float]
    bh_curve:         List[float]
    elapsed_s:        float


class HistoryResponse(BaseModel):
    ticker:   str
    dates:    List[str]
    prices:   List[float]
    volumes:  List[float]


class OptimizeResponse(BaseModel):
    tickers:          List[str]
    max_sharpe_w:     List[float]
    min_vol_w:        List[float]
    sim_returns:      List[float]
    sim_vols:         List[float]
    sim_sharpe:       List[float]
    max_sharpe_perf:  List[float]
    min_vol_perf:     List[float]
    corr_matrix:       List[List[float]]
    port_dates:        List[str]
    max_sharpe_equity: List[float]
    min_vol_equity:    List[float]
    equal_w_equity:    List[float]
    elapsed_s:        float
    asset_regimes:     Dict[str, str] = {}


class CustomPortfolioResponse(BaseModel):
    tickers:          List[str]
    weights:          List[float]
    exp_return_pct:   float
    volatility_pct:   float
    sharpe_ratio:     float
    efficiency_score: float
    equity_curve:     List[float]
    dates:            List[str]
    elapsed_s:        float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["ui"])
def root():
    from fastapi.responses import HTMLResponse
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>index.html not found</h1><p>Please build index.html in the project root.</p>",
            status_code=404
        )


@app.get("/predict", response_model=PredictionResponse, tags=["ml"])
def get_prediction(
    ticker:  str = Query(...,      description="Stock ticker, e.g. AAPL"),
    horizon: str = Query("daily",  description="daily | weekly | monthly"),
    years:   int = Query(5,        ge=2, le=10, description="Years of history"),
):
    """
    Return XGBoost price prediction + 80% CI for the given horizon.
    Horizon choices: daily (t+1), weekly (t+5), monthly (t+21).
    """
    t0 = time.perf_counter()
    try:
        res = _cached_predict(ticker.upper(), horizon, years)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception(f"Prediction error for {ticker}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    import math
    pred_ret_pct = (math.exp(res.pred_log_return) - 1) * 100

    return PredictionResponse(
        ticker           = res.ticker,
        horizon          = res.horizon_name,
        last_price       = round(res.last_price, 4),
        pred_price       = round(res.pred_price, 4),
        pred_return_pct  = round(pred_ret_pct, 4),
        ci_lower         = round(res.ci_lower_price, 4),
        ci_upper         = round(res.ci_upper_price, 4),
        hist_dates       = res.hist_dates,
        hist_prices      = [round(p, 4) for p in res.hist_prices],
        wf_dates         = res.wf_dates,
        wf_pred_prices   = [round(p, 4) for p in res.wf_pred_prices],
        wf_actual_prices = [round(p, 4) for p in res.wf_actual_prices],
        rmse             = round(res.rmse, 6),
        vol_21_pct       = round(res.vol_21 * 100, 4),
        elapsed_s        = round(time.perf_counter() - t0, 2),
        market_regime    = res.market_regime,
        shap_values      = {k: round(v, 6) for k, v in res.shap_values.items()},
        news             = res.news,
        mc_paths         = [[round(p, 4) for p in path] for path in res.mc_paths],
    )


@app.get("/optimize", response_model=OptimizeResponse, tags=["portfolio"])
def get_optimization(
    tickers: str = Query(...,      description="Comma-separated tickers, e.g. AAPL,MSFT,TSLA"),
    months:  int = Query(12,       ge=1, le=240, description="Number of months to simulate"),
    years:   int = Query(5,        ge=2, le=10),
    min_w:   float = Query(0.0,    ge=0.0, le=1.0, description="Minimum weight constraint per asset"),
    max_w:   float = Query(1.0,    ge=0.0, le=1.0, description="Maximum weight constraint per asset"),
):
    """
    Run Monte Carlo portfolio optimization based on predictions and covariance.
    """
    t0 = time.perf_counter()
    try:
        res = _cached_optimize(tickers, months, years, min_w, max_w)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception(f"Optimization error for {tickers}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return OptimizeResponse(
        tickers          = res.tickers,
        max_sharpe_w     = [round(w, 4) for w in res.max_sharpe_w],
        min_vol_w        = [round(w, 4) for w in res.min_vol_w],
        sim_returns      = [round(r, 4) for r in res.sim_returns],
        sim_vols         = [round(v, 4) for v in res.sim_vols],
        sim_sharpe       = [round(s, 4) for s in res.sim_sharpe],
        max_sharpe_perf  = [round(p, 4) for p in res.max_sharpe_perf],
        min_vol_perf     = [round(p, 4) for p in res.min_vol_perf],
        corr_matrix      = [[round(v, 4) for v in row] for row in res.corr_matrix],
        port_dates       = res.port_dates,
        max_sharpe_equity = [round(e, 4) for e in res.max_sharpe_equity],
        min_vol_equity   = [round(e, 4) for e in res.min_vol_equity],
        equal_w_equity   = [round(e, 4) for e in res.equal_w_equity],
        elapsed_s        = round(time.perf_counter() - t0, 2),
        asset_regimes    = res.asset_regimes,
    )


@app.get("/custom_portfolio", response_model=CustomPortfolioResponse, tags=["portfolio"])
def get_custom_portfolio(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,TSLA"),
    weights: str = Query(..., description="Comma-separated weights matching tickers, e.g. 0.4,0.3,0.3"),
    months:  int = Query(12,  ge=1, le=240, description="Simulation months"),
    years:   int = Query(5,   ge=2, le=10),
):
    """
    Evaluate user-defined custom portfolio weight allocation.
    Returns expected return %, volatility/risk %, Sharpe ratio, efficiency score, and equity curve.
    """
    t0 = time.perf_counter()
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
        
        if len(ticker_list) != len(weight_list):
            raise ValueError("Number of tickers must match number of weights.")
            
        def _fetch_and_train_custom(t: str):
            ohlcv = _cached_ohlcv(t, years)
            res = _cached_predict(t, "monthly", years)
            return t, ohlcv, res.pred_log_return

        ohlcv_dict = {}
        predictions = {}
        with ThreadPoolExecutor(max_workers=min(8, len(ticker_list))) as executor:
            for t, ohlcv, pred_lr in executor.map(_fetch_and_train_custom, ticker_list):
                ohlcv_dict[t] = ohlcv
                predictions[t] = pred_lr
            
        res = calculate_custom_portfolio(ohlcv_dict, predictions, weight_list, months)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception(f"Custom portfolio evaluation error for {tickers}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return CustomPortfolioResponse(
        tickers          = res.tickers,
        weights          = [round(w, 4) for w in res.weights],
        exp_return_pct   = res.exp_return_pct,
        volatility_pct   = res.volatility_pct,
        sharpe_ratio     = res.sharpe_ratio,
        efficiency_score = res.efficiency_score,
        equity_curve     = [round(e, 4) for e in res.equity_curve],
        dates            = res.dates,
        elapsed_s        = round(time.perf_counter() - t0, 2),
    )


@app.get("/backtest", response_model=BacktestResponse, tags=["backtest"])
def get_backtest(
    ticker:  str = Query(...,      description="Stock ticker"),
    horizon: str = Query("daily",  description="daily | weekly | monthly"),
    years:   int = Query(5,        ge=2, le=10),
):
    """
    Run long-or-flat backtest with transaction costs on OOS predictions.
    """
    t0 = time.perf_counter()
    try:
        res = _cached_backtest(ticker.upper(), horizon, years)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception(f"Backtest error for {ticker}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return BacktestResponse(
        ticker           = res.ticker,
        horizon          = res.horizon_name,
        total_return_pct = round(res.total_return_pct, 2),
        buy_hold_pct     = round(res.buy_hold_pct, 2),
        sharpe_ratio     = res.sharpe_ratio,
        max_drawdown_pct = res.max_drawdown_pct,
        n_trades         = res.n_trades,
        dates            = res.dates,
        equity_curve     = [round(v, 6) for v in res.equity_curve],
        bh_curve         = [round(v, 6) for v in res.bh_curve],
        elapsed_s        = round(time.perf_counter() - t0, 2),
    )


@app.get("/history", response_model=HistoryResponse, tags=["data"])
def get_history(
    ticker: str = Query(..., description="Stock ticker"),
    years:  int = Query(1,   ge=1, le=10),
):
    """Return raw OHLCV history for charting."""
    try:
        ohlcv = _cached_ohlcv(ticker.upper(), years)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = ohlcv.df.iloc[-252 * years :]
    return HistoryResponse(
        ticker  = ticker.upper(),
        dates   = [str(d.date()) for d in df.index],
        prices  = df["close"].round(4).tolist(),
        volumes = df["volume"].tolist(),
    )


@app.get("/debug", tags=["debug"])
def debug_yfinance():
    import traceback
    import requests
    import yfinance as yf
    
    output = {}
    
    # Test 1: Standard yfinance download (no session)
    try:
        df = yf.download("AAPL", period="1d", progress=False)
        output["test1_standard_yfinance"] = f"Success, shape: {df.shape}"
    except Exception as e:
        output["test1_standard_yfinance"] = f"Error: {str(e)}\n{traceback.format_exc()}"
        
    # Test 2: yfinance download with standard requests session + UA + verify=False (our new adapter method)
    try:
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        df = yf.download("AAPL", period="1d", progress=False, session=session)
        output["test2_new_adapter_yfinance"] = f"Success, shape: {df.shape}"
    except Exception as e:
        output["test2_new_adapter_yfinance"] = f"Error: {str(e)}\n{traceback.format_exc()}"
        
    # Test 3: Standard HTTP request (verify=True, no User-Agent)
    try:
        r = requests.get("https://query2.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1d", timeout=5)
        output["test3_http_no_ua_verify_true"] = f"Status: {r.status_code}, content length: {len(r.content)}"
    except Exception as e:
        output["test3_http_no_ua_verify_true"] = f"Error: {str(e)}\n{traceback.format_exc()}"

    # Test 4: New HTTP request (verify=False + User-Agent)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get("https://query2.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1d", timeout=5, verify=False, headers=headers)
        output["test4_http_ua_verify_false"] = f"Status: {r.status_code}, content length: {len(r.content)}"
    except Exception as e:
        output["test4_http_ua_verify_false"] = f"Error: {str(e)}\n{traceback.format_exc()}"

    return output
