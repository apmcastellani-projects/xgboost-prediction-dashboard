# Multi-Horizon XGBoost Predictive Dashboard
### End-to-End Time-Series Machine Learning Engine with Purged Walk-Forward Cross-Validation and FastAPI Backend

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Validation](https://img.shields.io/badge/Validation-Purged%20Walk--Forward-green.svg)]()

> **Author**: **Alessandro Castellani**  
> *Undergraduate background in Mathematics (Università dell'Insubria) | Graduate coursework in Applied Statistics & Data Science (Università Cattolica del Sacro Cuore)*  
> 📬 [alecaste041202@gmail.com](mailto:alecaste041202@gmail.com) | 🔗 [LinkedIn Profile](https://www.linkedin.com/in/alessandro-castellani-4905a7246-4905a7246/-4905a7246/) | 🐙 [GitHub Profile](https://github.com/apmcastellani-projects)

---

## 📌 Project Overview

This repository features an end-to-end quantitative machine learning pipeline and decision-support dashboard designed for time-series forecasting:
- **Predictive Engine**: Multi-horizon XGBoost price-return predictor (daily, weekly, monthly forecast horizons).
- **Leakage Prevention**: Strictly implements **purged walk-forward cross-validation** with embargo buffers, eliminating lookahead bias and temporal leakage.
- **RESTful Architecture**: High-performance backend powered by **FastAPI** with automatic OpenAPI documentation.
- **Interactive UI**: Standalone front-end dashboard (`index.html`) rendering interactive charts, confidence intervals, and backtesting telemetry.

---

## 🏗️ Architecture & Component Design

```
xgboost-prediction-dashboard/
├── ml_engine.py       # Feature engineering, XGBoost training, purged walk-forward validation & backtest
├── data_adapter.py    # Adapter pattern for real-time and historical data ingestion
├── api.py             # FastAPI REST endpoints (/predict, /backtest, /history)
├── index.html         # Interactive web dashboard
└── requirements.txt   # Core Python dependencies
```

### Key Methodological Highlights:
1. **Stationary Target Variable**: Predicts log-returns rather than raw price levels, preventing spurious correlations and scale leakage.
2. **Purged Walk-Forward Validation**: 5-fold expanding window with embargo periods to ensure test data independence.
3. **Empirical Confidence Intervals**: $\pm 1.96 	imes 	ext{OOS residual standard deviation}$ computed from out-of-sample prediction residuals.
4. **Realistic Backtesting Engine**: Incorporates transaction costs, realistic long-or-flat trade mechanics, and noise filtering ($	ext{pred\_return} > 0.5 	imes 	ext{rolling volatility}$).

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/apmcastellani-projects/xgboost-prediction-dashboard.git
cd xgboost-prediction-dashboard

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the FastAPI Server
```bash
uvicorn api:app --reload --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 3. Open the Dashboard
Simply open `index.html` in any modern web browser to interact with the visual dashboard.

---

## 📬 Contact & Opportunities

I am actively seeking **internship and analytical collaboration opportunities** across quantitative modeling, sports analytics, and data engineering.

- **Email**: [alecaste041202@gmail.com](mailto:alecaste041202@gmail.com)
- **LinkedIn**: [linkedin.com/in/alessandro-castellani-4905a7246](https://www.linkedin.com/in/alessandro-castellani-4905a7246-4905a7246/-4905a7246/)
- **GitHub**: [github.com/apmcastellani-projects](https://github.com/apmcastellani-projects)
