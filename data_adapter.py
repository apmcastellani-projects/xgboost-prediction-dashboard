# data_adapter.py
# ---------------------------------------------------------------------------
# Abstract data adapter + concrete implementations.
# To swap to Alpaca: implement AlpacaAdapter and change one line in api.py.
# ---------------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import urllib3
import warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


# ── Domain object ────────────────────────────────────────────────────────────

@dataclass
class OHLCV:
    """Normalised price/volume dataframe with a DatetimeIndex."""
    ticker: str
    df: pd.DataFrame          # columns: open, high, low, close, volume (lowercase)

    def validate(self) -> "OHLCV":
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"OHLCV missing columns: {missing}")
        if self.df.empty:
            raise ValueError(f"Empty dataframe for ticker '{self.ticker}'")
        return self


# ── Abstract interface ───────────────────────────────────────────────────────

class MarketDataAdapter(ABC):
    """All adapters must implement this contract."""

    @abstractmethod
    def fetch(self, ticker: str, years: int = 5) -> OHLCV:
        """Return OHLCV data for *ticker* covering the last *years* years."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


# ── YFinance adapter (default / prototype) ───────────────────────────────────

class YFinanceAdapter(MarketDataAdapter):
    """
    Free adapter using yfinance.
    No API key required.
    """

    def fetch(self, ticker: str, years: int = 5) -> OHLCV:
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=years)

        import requests
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            session=session,
        )

        if raw.empty:
            raise ValueError(f"yfinance returned no data for '{ticker}'")

        # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index   = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        return OHLCV(ticker=ticker, df=df).validate()

    def name(self) -> str:
        return "yfinance"


# ── Factory ──────────────────────────────────────────────────────────────────

def get_adapter(source: str = "yfinance") -> MarketDataAdapter:
    """
    Factory function.
    Usage:
        adapter = get_adapter("yfinance")
    """
    registry = {
        "yfinance": YFinanceAdapter,
    }
    if source not in registry:
        raise ValueError(f"Unknown adapter '{source}'. Choose from: {list(registry)}")
    return registry[source]()

