# 📉 Bitcoin Fall Prediction System

A machine learning system that predicts Bitcoin price falls using ensemble models with API and web dashboard.

## 🚀 Quick Start

### Local Development
```bash
# 1. Start API
cd api && python app_simple.py

# 2. Start Dashboard  
cd streamlit_app && streamlit run app.py

# 3. Access
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### Docker
```bash
# Start both services
docker-compose up --build

# Optional: Train new models
docker-compose --profile training up training
```

## 🏗️ Project Structure

```
bitcoin_SKEW/
├── api/                          # FastAPI backend
│   ├── app_simple.py            # Main API server
│   └── Dockerfile               # API container
├── streamlit_app/               # Streamlit dashboard
│   ├── app.py                   # Main dashboard
│   ├── requirements.txt         # Dependencies
│   └── Dockerfile               # Dashboard container
├── models/                      # Trained models
│   └── BTC_USD_latest/         # Latest Bitcoin model
├── generalized_fall_predictor.py # Core ML model
├── train_and_save_model.py     # Model training script
├── fetch_pretrained_models.py  # Deployment helper
├── docker-compose.yml          # Local development
├── render.yaml                 # Cloud deployment
└── requirements.txt            # API dependencies
```

## 🔮 Features

### 🤖 **Machine Learning**
- **3-Model Ensemble**: 5_30_30, 2_7_7, 4_15_15 configurations
- **Fall Detection**: Predicts significant Bitcoin price drops
- **Feature Engineering**: 160+ technical indicators
- **Uptrend Filtering**: Smart signal filtering during uptrends

### 🌐 **API Endpoints**
- `GET /predict/today` - Current prediction
- `GET /predict/history` - Historical analysis
- `GET /model/status` - Model information
- `GET /health` - System health

### 📊 **Dashboard**
- Real-time Bitcoin predictions
- Performance metrics and charts
- Historical backtesting
- Downloadable data (CSV)

### cURL
```bash
# Current prediction
curl "https://your-api.onrender.com/predict/today?ticker=BTC-USD"

# 30-day history
curl "https://your-api.onrender.com/predict/history?days=30"
```

## 🧪 Testing

```bash
# Test API
python test_simple_api.py

# Test Streamlit compatibility
python test_streamlit.py
```

## 📚 Key Files

- **`generalized_fall_predictor.py`** - Core ML model implementation
- **`app_simple.py`** - Simplified API for production
- **`streamlit_app/app.py`** - Web dashboard
- **`fetch_pretrained_models.py`** - Model deployment helper
- **`render.yaml`** - Cloud deployment configuration

## 🔄 Model Training

```bash
# Train new Bitcoin model
python train_and_save_model.py

# Or use Docker
docker-compose --profile training up training
```

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit      │───▶│    FastAPI       │───▶│   ML Models     │
│  Dashboard      │    │    Backend       │    │   (Ensemble)    │
│  (Frontend)     │    │  (Predictions)   │    │   + Features    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Production Ready

- ✅ **Containerized** with Docker
- ✅ **Cloud deployable** (Render/Streamlit Cloud)
- ✅ **Health checks** and monitoring
- ✅ **Pretrained models** included
- ✅ **Comprehensive testing**

---

**Built for Bitcoin fall prediction using machine learning ensemble methods**