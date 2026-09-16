# ml_engine.py
# ---------------------------------------------------------------------------
# Feature engineering, multi-horizon XGBoost models (t+1 / t+5 / t+21),
# purged walk-forward validation, confidence intervals, realistic backtest.
# ---------------------------------------------------------------------------

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

from xgboost import XGBRegressor

class SimpleScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit_transform(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        if isinstance(self.std, np.ndarray):
            self.std[self.std == 0.0] = 1.0
        elif self.std == 0.0:
            self.std = 1.0
        return (X_arr - self.mean) / self.std
    
    def transform(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean) / self.std

from data_adapter import OHLCV


# ── Constants ────────────────────────────────────────────────────────────────

HORIZONS: Dict[str, int] = {"daily": 1, "weekly": 5, "monthly": 21}

# Walk-forward parameters
MIN_TRAIN_BARS  = 252          # ~1 year minimum training window
PURGE_BARS      = 21           # embargo after each test fold (prevents leakage)
N_SPLITS        = 5            # number of WF folds

# Transaction costs
TRANSACTION_COST = 0.001       # 0.1% per trade (round-trip counted separately)

# Strategy threshold multiplier
K_THRESHOLD = 0.5              # enter long if pred_return > k * rolling_vol


# ── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a rich feature set to the OHLCV dataframe.
    All features are computed from past data only (no lookahead).
    """
    out = df.copy()
    c   = out["close"]
    v   = out["volume"]

    # ── Returns ─────────────────────────────────────────────────────────────
    out["log_ret"]    = np.log(c / c.shift(1))
    out["ret_1"]      = c.pct_change(1)
    out["ret_2"]      = c.pct_change(2)
    out["ret_3"]      = c.pct_change(3)
    out["ret_5"]      = c.pct_change(5)
    out["ret_10"]     = c.pct_change(10)
    out["ret_21"]     = c.pct_change(21)

    # ── Moving averages ──────────────────────────────────────────────────────
    for w in [10, 20, 50, 200]:
        out[f"sma_{w}"]      = c.rolling(w).mean()
        out[f"sma_{w}_ratio"] = c / out[f"sma_{w}"] - 1

    for w in [10, 20, 50]:
        out[f"ema_{w}"]      = c.ewm(span=w, adjust=False).mean()
        out[f"ema_{w}_ratio"] = c / out[f"ema_{w}"] - 1

    # ── Volatility ───────────────────────────────────────────────────────────
    out["vol_10"]  = out["log_ret"].rolling(10).std()
    out["vol_21"]  = out["log_ret"].rolling(21).std()
    out["vol_63"]  = out["log_ret"].rolling(63).std()
    out["vol_ratio"] = out["vol_10"] / (out["vol_21"] + 1e-8)

    # Garman-Klass Volatility (incorporates High, Low, Open, Close)
    log_hl = np.log(out["high"] / (out["low"] + 1e-8))
    log_co = np.log(out["close"] / (out["open"] + 1e-8))
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    out["vol_gk_21"] = np.sqrt(np.maximum(0, gk_var)).rolling(21).mean()

    # ATR (14-period)
    hl   = out["high"] - out["low"]
    hpc  = (out["high"] - out["close"].shift(1)).abs()
    lpc  = (out["low"]  - out["close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / c

    # Bollinger Band position
    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    out["bb_pos"] = (c - bb_mid) / (2 * bb_std + 1e-8)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["rsi_14_norm"] = out["rsi_14"] / 100.0

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_diff"]  = (macd - signal) / (c + 1e-8)
    out["macd_hist"]  = out["macd_diff"].diff()

    # ── Volume features ──────────────────────────────────────────────────────
    out["vol_sma_20"]   = v.rolling(20).mean()
    out["vol_ratio_20"] = v / (out["vol_sma_20"] + 1e-8)
    out["vol_log"]      = np.log1p(v)
    out["obv"]          = (np.sign(out["log_ret"]) * v).cumsum()
    out["obv_norm"]     = out["obv"] / (out["obv"].abs().rolling(63).max() + 1e-8)

    # ── Lagged features (prevent lookahead) ─────────────────────────────────
    for lag in [1, 2, 3, 5]:
        out[f"lag_ret_{lag}"]  = out["log_ret"].shift(lag)
        out[f"lag_vol_{lag}"]  = out["vol_10"].shift(lag)
        out[f"lag_rsi_{lag}"]  = out["rsi_14_norm"].shift(lag)

    # ── Calendar ─────────────────────────────────────────────────────────────
    out["day_of_week"]   = out.index.dayofweek / 4.0
    out["month"]         = out.index.month / 12.0
    out["quarter"]       = out.index.quarter / 4.0

    return out


FEATURE_COLS = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_21",
    "sma_10_ratio","sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "ema_10_ratio","ema_20_ratio","ema_50_ratio",
    "vol_10","vol_21","vol_63","vol_ratio","vol_gk_21",
    "atr_14","bb_pos",
    "rsi_14_norm","macd_diff","macd_hist",
    "vol_ratio_20","vol_log","obv_norm",
    "lag_ret_1","lag_ret_2","lag_ret_3","lag_ret_5",
    "lag_vol_1","lag_vol_2",
    "lag_rsi_1","lag_rsi_2",
    "day_of_week","month","quarter",
]


# ── Target construction ──────────────────────────────────────────────────────

def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Cumulative log-return over the next *horizon* bars.
    Uses shift(-horizon) — only safe when rows are NOT used for future peeking.
    The training code trims accordingly.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    # Sum of future log-returns
    target = log_ret.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return target


# ── Purged Walk-Forward Validation ───────────────────────────────────────────

@dataclass
class WFResult:
    oos_preds:    np.ndarray
    oos_actuals:  np.ndarray
    oos_index:    pd.DatetimeIndex
    oos_folds:    np.ndarray
    oos_residuals: np.ndarray = field(init=False)
    rmse:         float       = field(init=False)

    def __post_init__(self):
        self.oos_residuals = self.oos_actuals - self.oos_preds
        self.rmse = float(np.sqrt(np.mean(self.oos_residuals ** 2)))


class RidgeRegressor:
    """Pure NumPy L2-regularized Linear Regression for ensemble blending."""
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_features = X_arr.shape[1]
        A = X_arr.T @ X_arr + self.alpha * np.eye(n_features)
        b = X_arr.T @ y_arr
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_


def _xgb_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators      = 300,
        learning_rate     = 0.03,
        max_depth         = 4,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        reg_lambda        = 1.0,
        reg_alpha         = 0.1,
        objective         = "reg:pseudohubererror",
        early_stopping_rounds = 30,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
) -> Tuple[XGBRegressor, RidgeRegressor, SimpleScaler, WFResult]:
    """
    Purged walk-forward cross-validation with Early Stopping & Ensemble Blending.

    For overlapping targets (horizon > 1) we apply an embargo of `horizon`
    bars between the end of the training fold and the start of the test fold
    to avoid leakage from overlapping return windows.
    """
    n       = len(X)
    purge   = max(PURGE_BARS, horizon)   # embargo
    fold_sz = (n - MIN_TRAIN_BARS) // N_SPLITS

    all_preds   : List[float]          = []
    all_actuals : List[float]          = []
    all_idx     : List[pd.Timestamp]   = []
    all_folds   : List[int]            = []

    for fold in range(N_SPLITS):
        train_end   = MIN_TRAIN_BARS + fold * fold_sz
        test_start  = train_end + purge
        test_end    = min(test_start + fold_sz, n)

        if test_end <= test_start:
            continue

        X_tr, y_tr = X.iloc[:train_end],             y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end],    y.iloc[test_start:test_end]

        # Drop NaNs in training
        mask_tr = y_tr.notna() & X_tr.notna().all(axis=1)
        mask_te = y_te.notna() & X_te.notna().all(axis=1)
        if mask_tr.sum() < 100 or mask_te.sum() < 5:
            continue

        scaler = SimpleScaler()
        X_tr_s = scaler.fit_transform(X_tr[mask_tr])
        X_te_s = scaler.transform(X_te[mask_te])

        y_tr_vec = y_tr[mask_tr].to_numpy()

        # Split 80/20 for early stopping evaluation set
        val_sz = max(10, int(len(X_tr_s) * 0.2))
        X_tr_sub, y_tr_sub = X_tr_s[:-val_sz], y_tr_vec[:-val_sz]
        X_val_sub, y_val_sub = X_tr_s[-val_sz:], y_tr_vec[-val_sz:]

        mdl = _xgb_model()
        mdl.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False)

        ridge = RidgeRegressor(alpha=10.0).fit(X_tr_s, y_tr_vec)

        xgb_preds = mdl.predict(X_te_s)
        ridge_preds = ridge.predict(X_te_s)

        # 70/30 Ensemble blend
        preds = 0.7 * xgb_preds + 0.3 * ridge_preds

        all_preds.extend(preds.tolist())
        all_actuals.extend(y_te[mask_te].tolist())
        all_idx.extend(X_te[mask_te].index.tolist())
        all_folds.extend([fold] * len(preds))

    # Final model trained on ALL data (used for live prediction)
    mask_all  = y.notna() & X.notna().all(axis=1)
    scaler_f  = SimpleScaler()
    X_all_s   = scaler_f.fit_transform(X[mask_all])
    y_all_vec = y[mask_all].to_numpy()

    val_sz = max(10, int(len(X_all_s) * 0.2))
    final_mdl = _xgb_model()
    final_mdl.fit(X_all_s[:-val_sz], y_all_vec[:-val_sz], eval_set=[(X_all_s[-val_sz:], y_all_vec[-val_sz:])], verbose=False)
    final_ridge = RidgeRegressor(alpha=10.0).fit(X_all_s, y_all_vec)

    wf = WFResult(
        oos_preds   = np.array(all_preds),
        oos_actuals = np.array(all_actuals),
        oos_index   = pd.DatetimeIndex(all_idx),
        oos_folds   = np.array(all_folds),
    )
    return final_mdl, final_ridge, scaler_f, wf


# ── Prediction pipeline ──────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    ticker:          str
    horizon_name:    str
    horizon_bars:    int
    last_price:      float
    pred_log_return: float
    pred_price:      float
    ci_lower_price:  float
    ci_upper_price:  float
    hist_dates:      List[str]
    hist_prices:     List[float]
    wf_dates:        List[str]          # OOS dates for backtest chart
    wf_pred_prices:  List[float]
    wf_actual_prices: List[float]
    rmse:            float
    vol_21:          float              # current 21-day volatility
    market_regime:   str
    shap_values:     Dict[str, float]
    news:            List[Dict[str, Any]]
    mc_paths:        List[List[float]] = field(default_factory=list)


def predict(ohlcv: OHLCV, horizon_name: str = "daily") -> PredictionResult:
    """
    Full pipeline: features → train → OOS validation → live prediction.
    """
    import time
    horizon_name = horizon_name.lower()
    if horizon_name not in HORIZONS:
        raise ValueError(f"horizon_name must be one of {list(HORIZONS.keys())}")
    h = HORIZONS[horizon_name]

    df  = add_features(ohlcv.df)
    tgt = make_target(df, h)

    # Align
    valid_idx = df.index.intersection(tgt.dropna().index)
    X   = df.loc[valid_idx, FEATURE_COLS]
    y   = tgt.loc[valid_idx]

    # Drop rows with NaN features
    mask   = X.notna().all(axis=1) & y.notna()
    X, y   = X[mask], y[mask]

    if len(X) < MIN_TRAIN_BARS + 50:
        raise ValueError("Not enough data to train. Try a longer history or different ticker.")

    # Train + WF validation
    model, ridge_model, scaler, wf = walk_forward_validate(X, y, h)

    # ── Live prediction ──────────────────────────────────────────────────────
    last_row      = df[FEATURE_COLS].dropna().iloc[[-1]]
    last_row_s    = scaler.transform(last_row)
    xgb_pred_lr   = float(model.predict(last_row_s)[0])
    ridge_pred_lr = float(ridge_model.predict(last_row_s)[0])
    pred_lr       = 0.7 * xgb_pred_lr + 0.3 * ridge_pred_lr
    last_price    = float(ohlcv.df["close"].iloc[-1])
    pred_price    = last_price * np.exp(pred_lr)

    # 80% CI from OOS residual std
    resid_std   = float(np.std(wf.oos_residuals))
    ci_half     = 1.28155 * resid_std
    ci_lo       = last_price * np.exp(pred_lr - ci_half)
    ci_hi       = last_price * np.exp(pred_lr + ci_half)

    # Current volatility
    vol_21 = float(df["vol_21"].iloc[-1])

    # ── OOS series → convert to price for chart ───────────────────────────
    # Reconstruct cumulative prices from OOS log-return predictions
    wf_prices_pred   = []
    wf_prices_actual = []

    for i, dt in enumerate(wf.oos_index):
        base = float(ohlcv.df["close"].asof(dt))
        wf_prices_pred.append(base * np.exp(wf.oos_preds[i]))
        wf_prices_actual.append(base * np.exp(wf.oos_actuals[i]))

    # ── Regime Detection ───────────────────────────────────────────────────
    regime, _ = detect_market_regime(ohlcv.df)

    # ── SHAP Contributions ─────────────────────────────────────────────────
    import xgboost as xgb
    dmat = xgb.DMatrix(last_row_s, feature_names=FEATURE_COLS)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)[0]
    feat_contribs = {}
    for idx, feat in enumerate(FEATURE_COLS):
        feat_contribs[feat] = float(contribs[idx])
    sorted_contribs = sorted(feat_contribs.items(), key=lambda item: abs(item[1]), reverse=True)
    top_shap = dict(sorted_contribs[:5])

    # ── News Sentiment ─────────────────────────────────────────────────────
    import requests
    import yfinance as yf
    session = requests.Session()
    session.verify = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    ticker_obj = yf.Ticker(ohlcv.ticker, session=session)
    raw_news = getattr(ticker_obj, "news", [])
    news_list = []
    
    if raw_news:
        for item in raw_news[:8]:
            content = item.get("content", {})
            if content:
                title = content.get("title", "") or ""
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
                url_dict = content.get("clickThroughUrl", {})
                link = url_dict.get("url", "") if isinstance(url_dict, dict) else ""
                
                date_str = ""
                pub_date = content.get("pubDate", "")
                if pub_date:
                    try:
                        dt = pd.to_datetime(pub_date)
                        date_str = dt.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        date_str = str(pub_date)
            else:
                title = item.get("title", "") or ""
                publisher = item.get("publisher", "") or ""
                link = item.get("link", "") or ""
                ts = item.get("providerPublishTime", 0)
                date_str = ""
                if ts:
                    date_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))
            
            score = score_sentiment(title)
            news_list.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "date": date_str,
                "sentiment": score
            })

    # ── Monte Carlo Price Simulation ─────────────────────────────────────────
    hist_log_ret = np.log(ohlcv.df["close"] / ohlcv.df["close"].shift(1)).dropna().iloc[-252:]
    mc_paths = []
    num_paths = 100
    path_len = 21
    if h == 5:
        path_len = 10
    elif h == 1:
        path_len = 5

    if len(hist_log_ret) > 10:
        for _ in range(num_paths):
            rand_rets = np.random.choice(hist_log_ret, size=path_len, replace=True)
            daily_drift = pred_lr / h
            rand_rets = rand_rets - np.mean(rand_rets) + daily_drift
            cum_ret = np.exp(np.cumsum(rand_rets))
            path_prices = (last_price * cum_ret).tolist()
            path_prices.insert(0, last_price)
            mc_paths.append(path_prices)
    else:
        mc_paths = [[last_price] * (path_len + 1) for _ in range(num_paths)]

    return PredictionResult(
        ticker           = ohlcv.ticker,
        horizon_name     = horizon_name,
        horizon_bars     = h,
        last_price       = last_price,
        pred_log_return  = pred_lr,
        pred_price       = pred_price,
        ci_lower_price   = ci_lo,
        ci_upper_price   = ci_hi,
        hist_dates       = [str(d.date()) for d in ohlcv.df.index[-252:]],
        hist_prices      = ohlcv.df["close"].iloc[-252:].tolist(),
        wf_dates         = [str(d.date()) for d in wf.oos_index],
        wf_pred_prices   = wf_prices_pred,
        wf_actual_prices = wf_prices_actual,
        rmse             = wf.rmse,
        vol_21           = vol_21,
        market_regime    = regime,
        shap_values      = top_shap,
        news             = news_list,
        mc_paths         = mc_paths,
    )


# ── Backtest ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker:           str
    horizon_name:     str
    total_return_pct: float
    buy_hold_pct:     float
    sharpe_ratio:     float
    max_drawdown_pct: float
    n_trades:         int
    dates:            List[str]
    equity_curve:     List[float]
    bh_curve:         List[float]


def backtest(ohlcv: OHLCV, horizon_name: str = "daily") -> BacktestResult:
    """
    Long-or-flat strategy on OOS predictions.
    Entry: pred_return > K * current_vol_21
    Costs: TRANSACTION_COST per entry AND exit leg.
    """
    horizon_name = horizon_name.lower()
    h = HORIZONS[horizon_name]

    df  = add_features(ohlcv.df)
    tgt = make_target(df, h)

    valid_idx = df.index.intersection(tgt.dropna().index)
    X   = df.loc[valid_idx, FEATURE_COLS]
    y   = tgt.loc[valid_idx]
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    _, _, scaler, wf = walk_forward_validate(X, y, h)

    # Align predictions with price series
    # daily_ret drives equity/bh updates; pred drives entry signal only
    oos_df = pd.DataFrame({
        "pred":      wf.oos_preds,
        "actual":    wf.oos_actuals,
        "fold":      wf.oos_folds,
        "vol_21":    df["vol_21"].reindex(wf.oos_index),
    }, index=wf.oos_index).dropna()

    if len(oos_df) < 10:
        raise ValueError("Too few OOS samples for backtest.")

    # Strategy signals
    threshold       = K_THRESHOLD * oos_df["vol_21"]
    oos_df["signal"] = (oos_df["pred"] > threshold).astype(int)

    # Equity curve and Buy & Hold curve
    # 1. Buy & Hold: simple ratio of the stock price normalized to 1.0 at start
    first_date = oos_df.index[0]
    initial_price = float(ohlcv.df["close"].asof(first_date))
    bh_curve = [float(ohlcv.df["close"].asof(dt)) / initial_price for dt in oos_df.index]

    # 2. Strategy: compound returns per fold, going flat at fold boundaries
    equity = 1.0
    equity_curve = [equity]
    n_trades = 0
    position = 0 # 0 = flat, 1 = long

    for i in range(1, len(oos_df)):
        prev_dt = oos_df.index[i - 1]
        curr_dt = oos_df.index[i]
        
        prev_signal = int(oos_df["signal"].iloc[i - 1])
        curr_signal = int(oos_df["signal"].iloc[i])
        
        prev_fold = int(oos_df["fold"].iloc[i - 1])
        curr_fold = int(oos_df["fold"].iloc[i])
        
        if prev_fold == curr_fold:
            # Within the same fold, standard continuation
            # Daily return from prev_dt to curr_dt
            p_prev = float(ohlcv.df["close"].asof(prev_dt))
            p_curr = float(ohlcv.df["close"].asof(curr_dt))
            daily_ret = p_curr / p_prev
            
            # Entry cost
            if position == 0 and curr_signal == 1:
                equity *= (1 - TRANSACTION_COST)
                n_trades += 1
                position = 1
            # Exit cost
            elif position == 1 and curr_signal == 0:
                equity *= (1 - TRANSACTION_COST)
                position = 0
                
            # Accrue return
            if position == 1:
                equity *= daily_ret
        else:
            # We are crossing a fold boundary!
            # If we were holding, we sell at the end of the previous fold
            if position == 1:
                equity *= (1 - TRANSACTION_COST)
                position = 0
            
            # During the embargo gap, we are flat, so equity doesn't change.
            # At the start of the new fold (curr_dt), if signal is 1, we buy.
            if curr_signal == 1:
                equity *= (1 - TRANSACTION_COST)
                n_trades += 1
                position = 1
                
        equity_curve.append(equity)

    equity_arr = np.array(equity_curve)

    # Max drawdown
    peak    = np.maximum.accumulate(equity_arr)
    dd      = (equity_arr - peak) / peak
    max_dd  = float(dd.min()) * 100

    # Sharpe (annualised, daily log-returns of equity)
    log_rets = np.diff(np.log(equity_arr + 1e-10))
    sharpe   = float(
        np.mean(log_rets) / (np.std(log_rets) + 1e-10) * np.sqrt(252)
    ) if len(log_rets) > 1 else 0.0

    dates = [str(d.date()) for d in oos_df.index]

    return BacktestResult(
        ticker           = ohlcv.ticker,
        horizon_name     = horizon_name,
        total_return_pct = (equity - 1) * 100,
        buy_hold_pct     = (bh_curve[-1] - 1) * 100,
        sharpe_ratio     = round(sharpe, 3),
        max_drawdown_pct = round(max_dd, 2),
        n_trades         = n_trades,
        dates            = dates,
        equity_curve     = equity_curve,
        bh_curve         = bh_curve,
    )


# ── Regime Detection (GMM) ────────────────────────────────────────────────────

# ── Regime Detection (Pure NumPy/Pandas) ──────────────────────────────────────

def detect_market_regime(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Classifies the current market regime using a pure NumPy/Pandas algorithm
    trained on rolling 21-day returns and rolling 21-day annualized volatility.
    Requires NO external dependencies like scikit-learn.
    """
    data = pd.DataFrame(index=df.index)
    data["ret_21"] = df["close"].pct_change(21)
    data["vol_21"] = df["close"].pct_change().rolling(21).std() * np.sqrt(252)
    
    data_clean = data.dropna()
    if len(data_clean) < 50:
        return "Unknown", ["Unknown"] * len(df)
        
    vol_mean = data_clean["vol_21"].mean()
    vol_std = data_clean["vol_21"].std() + 1e-8

    regimes_list = []
    for idx, row in data_clean.iterrows():
        r = row["ret_21"]
        v = row["vol_21"]
        
        if r > 0.015 and v <= vol_mean + 0.3 * vol_std:
            reg = "Bull Market"
        elif r < -0.015:
            reg = "Bear Market"
        else:
            reg = "Sideways / High Volatility"
        regimes_list.append(reg)

    current_regime = regimes_list[-1]
    all_regimes = ["Unknown"] * (len(df) - len(regimes_list)) + regimes_list
    return current_regime, all_regimes


# ── Sentiment Analysis Lexicon ────────────────────────────────────────────────

FINANCIAL_LEXICON = {
    "growth": 0.8, "profit": 0.8, "bullish": 0.9, "buy": 0.7, "upgrade": 0.9,
    "beat": 0.8, "gains": 0.7, "gain": 0.7, "surge": 0.8, "rise": 0.6,
    "higher": 0.5, "jump": 0.7, "boost": 0.7, "strong": 0.6, "outperform": 0.9,
    "positive": 0.7, "success": 0.7, "exceeds": 0.8, "exceed": 0.8, "beating": 0.8,
    "soar": 0.8, "soars": 0.8, "record": 0.7, "lead": 0.5, "up": 0.4,
    "loss": -0.8, "miss": -0.8, "bearish": -0.9, "sell": -0.7, "downgrade": -0.9,
    "fall": -0.6, "decline": -0.6, "drop": -0.7, "lower": -0.5, "plummet": -0.8,
    "slump": -0.8, "weak": -0.6, "underperform": -0.9, "negative": -0.7,
    "failure": -0.8, "missed": -0.8, "missing": -0.8, "crash": -0.9, "warning": -0.7,
    "down": -0.4, "debt": -0.5, "worry": -0.6, "concern": -0.6, "lawsuit": -0.7,
    "sink": -0.7, "sinks": -0.7
}

def score_sentiment(headline: str) -> float:
    """
    Calculates a sentiment score from -1.0 to 1.0 based on financial keywords.
    """
    import re
    words = re.findall(r'\b\w+\b', headline.lower())
    score = 0.0
    count = 0
    for word in words:
        if word in FINANCIAL_LEXICON:
            score += FINANCIAL_LEXICON[word]
            count += 1
    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / count))


# ── Multivariate DCC-APARCH Volatility & Covariance Model ─────────────────────

def estimate_multivariate_dcc_aparch_cov(returns_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 1-step ahead Multivariate Dynamic Conditional Covariance Matrix H_{t+1}
    using univariate APARCH/GJR-GARCH models for asset variances (capturing leverage effect and fat tails)
    and Engle's Dynamic Conditional Correlation (DCC) for time-varying correlations.
    """
    tickers = returns_df.columns
    N = len(tickers)
    T = len(returns_df)
    
    if T < 50:
        return returns_df.cov().values, np.zeros((T, N))
        
    cond_vols = np.zeros((T, N))
    std_resids = np.zeros((T, N))
    
    # 1. Fit Univariate APARCH / GJR-GARCH per asset
    for i, tk in enumerate(tickers):
        series = returns_df[tk] * 100.0  # Scale to percentages for GARCH stability
        try:
            from arch import arch_model
            am = arch_model(series, vol='Garch', p=1, o=1, q=1, dist='t')
            res = am.fit(disp='off', options={'maxiter': 50})
            cond_vols[:, i] = res.conditional_volatility / 100.0  # Rescale back
            resids = res.resid.values / 100.0
            std_resids[:, i] = resids / np.maximum(1e-6, cond_vols[:, i])
        except Exception:
            # Robust fallback to rolling standard deviation
            vol = series.rolling(21).std().fillna(series.std()).values / 100.0
            cond_vols[:, i] = vol
            std_resids[:, i] = (series.values / 100.0 - series.mean() / 100.0) / np.maximum(1e-6, vol)
            
    # 2. Dynamic Conditional Correlation (DCC) estimation
    Q_bar = np.cov(std_resids.T)
    Q_t = Q_bar.copy()
    a_opt, b_opt = 0.05, 0.90
    
    for t in range(T):
        e_t = std_resids[t, :].reshape(-1, 1)
        Q_t = (1 - a_opt - b_opt) * Q_bar + a_opt * (e_t @ e_t.T) + b_opt * Q_t
        
    q_diag = np.sqrt(np.maximum(1e-6, np.diag(Q_t)))
    R_t = Q_t / np.outer(q_diag, q_diag)
    
    latest_vols = cond_vols[-1, :]
    D_t = np.diag(latest_vols)
    H_t = D_t @ R_t @ D_t
    return H_t, cond_vols


# ── Portfolio Optimization ────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    tickers:         List[str]
    max_sharpe_w:    List[float]
    min_vol_w:       List[float]
    sim_returns:     List[float]
    sim_vols:        List[float]
    sim_sharpe:      List[float]
    max_sharpe_perf: Tuple[float, float, float]
    min_vol_perf:    Tuple[float, float, float]
    corr_matrix:     List[List[float]] = field(default_factory=list)
    port_dates:      List[str] = field(default_factory=list)
    max_sharpe_equity: List[float] = field(default_factory=list)
    min_vol_equity:  List[float] = field(default_factory=list)
    equal_w_equity:  List[float] = field(default_factory=list)
    asset_regimes:   Dict[str, str] = field(default_factory=dict)


@dataclass
class CustomPortfolioResult:
    tickers:          List[str]
    weights:          List[float]
    exp_return_pct:   float
    volatility_pct:   float
    sharpe_ratio:     float
    efficiency_score: float
    equity_curve:     List[float]
    dates:            List[str]


def optimize_portfolio(ohlcv_dict: Dict[str, OHLCV], predictions: Dict[str, float], months: int, min_w: float = 0.0, max_w: float = 1.0) -> OptimizationResult:
    """
    Runs a Dirichlet-sampled portfolio optimization using Multivariate DCC-APARCH
    conditional covariance matrix, scaled to the user-specified number of months.
    """
    tickers = list(ohlcv_dict.keys())
    num_assets = len(tickers)
    
    # 1. Align historical daily log-returns
    returns_df = pd.DataFrame()
    for t in tickers:
        close_prices = ohlcv_dict[t].df["close"]
        returns_df[t] = np.log(close_prices / close_prices.shift(1))
        
    returns_df = returns_df.dropna()
    
    # 2. Compute Multivariate DCC-APARCH Conditional Covariance Matrix
    H_daily, _ = estimate_multivariate_dcc_aparch_cov(returns_df)
    cov_matrix = H_daily * (21 * months)
    corr_matrix = returns_df.corr().values.tolist()
    
    # Expected returns for M months
    exp_returns = np.array([predictions[t] * months for t in tickers])
    rf = 0.02 * (months / 12.0)
    
    # Weight constraints check
    if min_w * num_assets > 1.0 or max_w * num_assets < 1.0:
        min_w, max_w = 0.0, 1.0
        
    # Dirichlet sampling for 500 visual scatter points
    num_portfolios = 500
    dirichlet_weights = np.random.dirichlet(np.ones(num_assets), size=num_portfolios)
    
    weights_record = []
    for w in dirichlet_weights:
        w_clipped = np.clip(w, min_w, max_w)
        w_norm = w_clipped / np.sum(w_clipped)
        weights_record.append(w_norm)
        
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        w = weights_record[i]
        p_ret = np.sum(w * exp_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        p_sharpe = (p_ret - rf) / (p_vol + 1e-10)
        
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_sharpe
        
    # Max Sharpe
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_w = weights_record[max_sharpe_idx].tolist()
    max_sharpe_perf = (float(results[0, max_sharpe_idx]), float(results[1, max_sharpe_idx]), float(results[2, max_sharpe_idx]))
    
    # Min Volatility
    min_vol_idx = np.argmin(results[1])
    min_vol_w = weights_record[min_vol_idx].tolist()
    min_vol_perf = (float(results[0, min_vol_idx]), float(results[1, min_vol_idx]), float(results[2, min_vol_idx]))
    
    # Historical portfolio equity curves
    port_dates = [str(d.date()) for d in returns_df.index]
    max_sharpe_equity = np.exp(returns_df.dot(np.array(max_sharpe_w)).cumsum()).tolist()
    min_vol_equity = np.exp(returns_df.dot(np.array(min_vol_w)).cumsum()).tolist()
    
    equal_w = np.array([1.0 / num_assets] * num_assets)
    equal_w_equity = np.exp(returns_df.dot(equal_w).cumsum()).tolist()
    
    return OptimizationResult(
        tickers = tickers,
        max_sharpe_w = max_sharpe_w,
        min_vol_w = min_vol_w,
        sim_returns = results[0].tolist(),
        sim_vols = results[1].tolist(),
        sim_sharpe = results[2].tolist(),
        max_sharpe_perf = max_sharpe_perf,
        min_vol_perf = min_vol_perf,
        corr_matrix = corr_matrix,
        port_dates = port_dates,
        max_sharpe_equity = max_sharpe_equity,
        min_vol_equity = min_vol_equity,
        equal_w_equity = equal_w_equity,
    )


def calculate_custom_portfolio(
    ohlcv_dict: Dict[str, OHLCV],
    predictions: Dict[str, float],
    weights: List[float],
    months: int = 12
) -> CustomPortfolioResult:
    """
    Evaluates a user-defined custom portfolio weight allocation.
    Calculates expected return, risk/volatility, custom Sharpe ratio,
    efficiency score relative to the optimal Max Sharpe allocation, and historical growth curve.
    """
    tickers = list(ohlcv_dict.keys())
    w = np.array(weights, dtype=np.float64)
    if np.sum(w) > 0:
        w /= np.sum(w)
    else:
        w = np.array([1.0 / len(tickers)] * len(tickers))
    
    returns_df = pd.DataFrame()
    for t in tickers:
        close_prices = ohlcv_dict[t].df["close"]
        returns_df[t] = np.log(close_prices / close_prices.shift(1))
    returns_df = returns_df.dropna()
    
    H_daily, _ = estimate_multivariate_dcc_aparch_cov(returns_df)
    cov_matrix = H_daily * (21 * months)
    exp_returns = np.array([predictions[t] * months for t in tickers])
    rf = 0.02 * (months / 12.0)
    
    p_ret = float(np.sum(w * exp_returns))
    p_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
    p_sharpe = float((p_ret - rf) / (p_vol + 1e-10))
    
    # Calculate Max Sharpe optimal for efficiency score comparison
    opt = optimize_portfolio(ohlcv_dict, predictions, months)
    max_sharpe_val = max(0.001, opt.max_sharpe_perf[2])
    efficiency_score = min(100.0, max(0.0, (p_sharpe / max_sharpe_val) * 100.0))
    
    port_dates = [str(d.date()) for d in returns_df.index]
    custom_equity = np.exp(returns_df.dot(w).cumsum()).tolist()
    
    return CustomPortfolioResult(
        tickers = tickers,
        weights = w.tolist(),
        exp_return_pct = round(p_ret * 100.0, 2),
        volatility_pct = round(p_vol * 100.0, 2),
        sharpe_ratio = round(p_sharpe, 3),
        efficiency_score = round(efficiency_score, 1),
        equity_curve = custom_equity,
        dates = port_dates
    )
