# 📊 Kenya Food Price Inflation Tracker

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A comprehensive data science project monitoring and forecasting food price inflation in Kenya using machine learning and time series analysis.**

🔗 **Live Dashboard**: [Coming Soon - Streamlit Cloud]  
👤 **Portfolio**: [https://muemastephenportfolio.netlify.app/](https://muemastephenportfolio.netlify.app/)  
📂 **Repository**: [https://github.com/Kaks753/food-inflation](https://github.com/Kaks753/food-inflation)

---

## 🎯 Project Overview

This project analyzes and forecasts food prices in Kenya using **15 years of data** (2006-2021) from the World Food Programme (WFP). It provides actionable insights for:
- **Policymakers**: Strategic planning and intervention timing
- **Farmers**: Optimal harvest and selling decisions
- **Consumers**: Budget planning and bulk buying strategies
- **Traders**: Arbitrage opportunities and inventory management

### 📈 Key Findings

| Metric | Value |
|--------|-------|
| **Data Period** | 2006-2021 (15 years) |
| **Records** | 8,884 observations |
| **Commodities** | 34 tracked items |
| **Markets** | 45 locations across 6 regions |
| **Maize Price Growth** | +45% (CAGR 2.5%) |
| **Seasonal Swing** | 15-20 KES/kg |
| **Regional Gap** | 20-30 KES/kg |
| **Model Accuracy** | 6-9% MAPE (ensemble) |

---

## 🚀 Features

### ✅ **Data Pipeline**
- ✅ Automated data cleaning and preprocessing
- ✅ Outlier detection using IQR method
- ✅ Feature engineering (lags, rolling stats, seasonality)
- ✅ Time series validation framework

### ✅ **Analysis & Modeling**
- ✅ **Exploratory Data Analysis** (trends, volatility, seasonality)
- ✅ **Prophet** forecasting (Facebook's time series model)
- ✅ **SARIMA** (Seasonal ARIMA) with auto-parameter tuning
- ✅ **XGBoost & LightGBM** (gradient boosting models)
- ✅ **Ensemble** approach for improved accuracy

### ✅ **Interactive Dashboard**
- ✅ **Overview**: Dataset summary and key metrics
- ✅ **Price Trends**: Time series visualization by commodity
- ✅ **Regional Analysis**: Geographic price comparisons
- ✅ **Inflation Calculator**: YoY inflation computation
- ✅ **Forecasts**: 12-month price predictions with confidence intervals
- ✅ **Insights**: Actionable recommendations for stakeholders

---

## 📁 Project Structure

```
food-inflation/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── raw/                        # Original WFP data
│   │   └── wfp_food_prices_kenya_full.csv
│   ├── clean/                      # Processed datasets
│   │   ├── wfp_kenya_clean.csv
│   │   ├── wfp_core_staples.csv
│   │   ├── wfp_monthly_avg.csv
│   │   ├── maize_features.csv
│   │   └── maize_features_ml.csv
│   └── archive/                    # Unused datasets
│
├── notebooks/                      # Jupyter notebooks (analysis pipeline)
│   ├── 01_data_cleaning_and_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04a_prophet_forecasting.ipynb
│   ├── 04b_sarima_forecasting.ipynb
│   ├── 04c_ml_forecasting.ipynb
│   ├── 04d_model_comparison.ipynb
│   └── 05_insights_and_recommendations.ipynb
│
├── models/
│   └── trained/                    # Saved models and forecasts
│       ├── prophet_maize_model.pkl
│       ├── sarima_maize_model.pkl
│       ├── xgboost_maize_model.pkl
│       ├── prophet_forecast.csv
│       ├── sarima_forecast.csv
│       ├── xgboost_forecast.csv
│       └── model_comparison.csv
│
├── src/                            # Python modules (reusable code)
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   └── download_food_data.py
│   └── utils/
│       └── helpers.py
│
└── reports/                        # Analysis outputs
    ├── eda_summary_statistics.csv
    └── figures/
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Kaks753/food-inflation.git
cd food-inflation
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Notebooks (Optional - to retrain models)

Open Jupyter and run notebooks in order:

```bash
jupyter notebook
```

**Notebook Sequence**:
1. `01_data_cleaning_and_preparation.ipynb` → Cleans raw data
2. `02_exploratory_data_analysis.ipynb` → Generates insights
3. `03_feature_engineering.ipynb` → Creates model features
4. `04a_prophet_forecasting.ipynb` → Trains Prophet model
5. `04b_sarima_forecasting.ipynb` → Trains SARIMA model
6. `04c_ml_forecasting.ipynb` → Trains XGBoost/LightGBM
7. `04d_model_comparison.ipynb` → Compares all models
8. `05_insights_and_recommendations.ipynb` → Synthesis

### 5. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at **http://localhost:8501**

---

## 📊 Model Performance

| Model | MAE | RMSE | MAPE | Notes |
|-------|-----|------|------|-------|
| **Prophet** | ~4.2 | ~5.8 | 8-12% | Best for trend + seasonality |
| **SARIMA** | ~3.8 | ~5.2 | 7-10% | Strong statistical foundation |
| **XGBoost** | ~3.2 | ~4.5 | 6-9% | **Best overall performance** |
| **LightGBM** | ~3.4 | ~4.7 | 6-9% | Fast training, good accuracy |
| **Ensemble** | ~3.0 | ~4.2 | **~6%** | **Recommended for production** |

*Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)*

---

## 💡 Key Insights

### 1. **Price Trends** 📈
- Maize prices rose **45%** from 2006 to 2021 (CAGR 2.5%)
- Peak price: **100 KES/kg** during 2017 drought
- Average price: **48 KES/kg**

### 2. **Seasonality** 🌾
- **Harvest months** (Oct-Dec): Prices drop 15-20%
- **Lean season** (Feb-May): Prices spike 15-20%
- Swing: **15-20 KES/kg**

### 3. **Regional Variation** 🗺️
- **Nairobi & Coast**: 50-55 KES/kg (higher, urban demand)
- **Rift Valley**: 40-45 KES/kg (lower, production region)
- Gap: **20-30 KES/kg**

### 4. **Volatility** ⚡
- Coefficient of variation: **~35%**
- Drought years (2011, 2017) show highest spikes
- Imported goods (rice, oil) more stable

---

## 🎯 Recommendations

### For Policymakers 🏛️
1. Build **strategic grain reserves** (3-6 month buffer)
2. Improve **transport infrastructure** (Rift Valley → Nairobi/Coast)
3. Implement **early warning system** with price monitoring
4. Time **subsidies** to lean season (Feb-May)

### For Farmers 👨‍🌾
1. **Store & sell** during lean season (+15-20% premium)
2. **Diversify crops** to reduce volatility exposure
3. Use **price forecasts** for planting decisions
4. Form **cooperatives** to reduce costs

### For Consumers 🏘️
1. **Bulk buy** during harvest (15-20% savings)
2. Invest in **proper storage** (3-6 month supply)
3. Budget for **+3-7% annual inflation**
4. Consider **substitute staples** when prices spike

### For Traders 🏢
1. **Arbitrage** regional gaps (20-30 KES/kg)
2. **Stock** during harvest, sell during lean season
3. Use **forecasts** to hedge commodity risk
4. **Optimize logistics** to reduce price gaps

---

## 📖 Data Sources

1. **World Food Programme (WFP)** - Kenya Food Prices (2006-2021)
   - Source: [HDX - Global Food Prices](https://data.humdata.org/dataset/global-wfp-food-prices)
   - Records: 8,884 observations
   - Coverage: 34 commodities, 45 markets, 6 regions

2. **FAO Food Price Index** (Reference)
   - Source: [FAO FPMA Tool](https://fpma.fao.org/)

3. **Kenya National Bureau of Statistics** (Supplementary)
   - Source: [KNBS](https://www.knbs.or.ke/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Stephen Muema**  
- Portfolio: [https://muemastephenportfolio.netlify.app/](https://muemastephenportfolio.netlify.app/)  
- GitHub: [@Kaks753](https://github.com/Kaks753)  
- LinkedIn: [Connect with me](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- **World Food Programme (WFP)** for providing comprehensive food price data
- **FAO** for global food security insights
- **Streamlit** for the amazing dashboard framework
- **Facebook Prophet**, **statsmodels**, **XGBoost** teams for excellent ML libraries

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: your.email@example.com
- **GitHub Issues**: [Open an issue](https://github.com/Kaks753/food-inflation/issues)

---

## 🔮 Roadmap

### Phase 1: MVP ✅ (Completed)
- [x] Data cleaning and EDA
- [x] Feature engineering
- [x] Model training (Prophet, SARIMA, XGBoost)
- [x] Streamlit dashboard
- [x] Documentation

### Phase 2: Enhancements 🚧 (In Progress)
- [ ] Add 2022-2024 data
- [ ] Multi-commodity forecasting
- [ ] Geospatial analysis with interactive maps
- [ ] Automated data updates

### Phase 3: Advanced Features 📋 (Planned)
- [ ] LSTM deep learning models
- [ ] Probabilistic forecasting
- [ ] Mobile app (SMS alerts)
- [ ] API endpoints
- [ ] Real-time monitoring

---

**⭐ If you find this project useful, please consider giving it a star!**

