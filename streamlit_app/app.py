"""
Streamlit app for Bitcoin Fall Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List
import os

# API configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Fall Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .signal-positive {
        color: #00cc00;
        font-weight: bold;
    }
    .signal-negative {
        color: #cc0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def fetch_prediction(ticker: str = 'BTC-USD'):
    """Fetch current prediction from API"""
    try:
        response = requests.get(f"{API_URL}/predict/today", params={"ticker": ticker})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching prediction: {e}")
        return None


def fetch_history(ticker: str = 'BTC-USD', days: int = 30):
    """Fetch historical predictions from API"""
    try:
        response = requests.get(
            f"{API_URL}/predict/history", 
            params={"ticker": ticker, "days": days}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return None


def fetch_model_status():
    """Fetch model status from API"""
    try:
        response = requests.get(f"{API_URL}/model/status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching model status: {e}")
        return None


# Training functions removed - using pretrained models only


def create_signal_chart(df: pd.DataFrame):
    """Create interactive chart with signals"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add buy signals
    buy_signals = df[df['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['close'],
            mode='markers',
            name='In Market',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ))
    
    # Add exit signals
    exit_signals = df[df['signal'] == 0]
    if not exit_signals.empty:
        fig.add_trace(go.Scatter(
            x=exit_signals['date'],
            y=exit_signals['close'],
            mode='markers',
            name='Out of Market',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ))
    
    fig.update_layout(
        title="Price and Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600
    )
    
    return fig


def create_returns_chart(df: pd.DataFrame):
    """Create cumulative returns chart"""
    # Calculate cumulative returns
    df['cum_return'] = (1 + df['return']).cumprod() - 1
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    fig = go.Figure()
    
    # Add strategy returns
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cum_strategy_return'] * 100,
        mode='lines',
        name='Strategy',
        line=dict(color='green', width=2)
    ))
    
    # Add benchmark returns
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cum_return'] * 100,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig


# Main app
def check_api_health():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    st.title("📉 Bitcoin Fall Prediction Dashboard")
    
    # API Health Check
    api_healthy = check_api_health()
    if not api_healthy:
        st.error(f"❌ Cannot connect to API at {API_URL}")
        st.info("Please check if the API is running and the URL is correct.")
        st.stop()
    else:
        st.success(f"✅ Connected to API at {API_URL}")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Fixed to BTC only
        ticker = "BTC-USD"
        st.info("🪙 Bitcoin (BTC-USD) Model Only")
        st.caption("This dashboard uses a pretrained Bitcoin-specific model")
        
        # History period
        history_days = st.slider(
            "History Period (days)",
            min_value=7,
            max_value=365,
            value=30
        )
        
        # Model information
        st.header("Model Information")
        model_status = fetch_model_status()
        if model_status and model_status['loaded']:
            # Check if this might be a demo model
            is_demo = (model_status.get('model_path', '').find('demo') != -1 or 
                      model_status.get('features_count', 0) == 30 and 
                      model_status.get('training_date', '').startswith('202'))
            
            if is_demo:
                st.warning("📋 Demo Model Active")
                st.info("Using simulated predictions. Train a real model for production use.")
            else:
                st.success("✅ Real Model Loaded")
            
            st.text(f"Ticker: {model_status['ticker']}")
            st.text(f"Trained: {model_status['training_date'][:8] if model_status['training_date'] else 'Unknown'}")
            st.text(f"Data until: {model_status['data_end_date']}")
            st.text(f"Features: {model_status.get('features_count', 'N/A')}")
            st.text(f"Version: {model_status.get('model_version', 'N/A')[:8]}")
        else:
            st.error("❌ No model loaded")
            st.info("The API may be starting up. Please refresh in a moment.")
        
        # Using Pretrained BTC Model Only
        st.header("Bitcoin Model")
        st.info("🪙 Pretrained Bitcoin Fall Predictor")
        st.markdown("""
        **Model Features:**
        • Trained on Bitcoin price data
        • 3-model ensemble prediction
        • Uptrend filtering capability
        • Historical backtesting available
        """)
        
        st.markdown("---")
        st.markdown("**📖 Documentation**")
        st.markdown("• [API Docs](http://localhost:8000/docs)")
        st.markdown("• [GitHub Repository](#)")
        st.markdown("• [Deployment Guide](#)")
    
    # Main content
    # Current prediction section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Current Prediction")
    with col2:
        if st.button("🔄 Refresh", help="Refresh predictions and data"):
            st.rerun()
    
    prediction = fetch_prediction(ticker)
    
    if prediction:
        # Check if this is a demo mode prediction
        metadata = prediction.get('metadata', {})
        if metadata.get('model_version') and len(metadata.get('target_configs', [])) == 3:
            # Check if using demo mode by looking at response structure
            individual_signals = prediction.get('individual_signals', {})
            if len(individual_signals) == 3:
                is_demo_prediction = any('demo' in str(metadata).lower() for key in metadata)
                if is_demo_prediction:
                    st.info("📋 Demo Mode: Predictions are simulated for demonstration purposes")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Date",
                prediction['date']
            )
        
        with col2:
            signal_class = "signal-positive" if prediction['signal_with_uptrend'] == 1 else "signal-negative"
            st.metric(
                "Signal (with uptrend)",
                "IN MARKET" if prediction['signal_with_uptrend'] == 1 else "OUT OF MARKET"
            )
            st.markdown(f"<p class='{signal_class}'>Ensemble Score: {prediction['ensemble_score']:.3f}</p>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.metric(
                "Signal (no filter)",
                f"{prediction['signal_without_uptrend']:.3f}"
            )
        
        with col4:
            uptrend_status = "Active" if prediction['uptrend_active'] else "Inactive"
            st.metric(
                "Uptrend Status",
                uptrend_status
            )
        
        # Individual model signals
        st.subheader("Individual Model Signals")
        
        col1, col2, col3 = st.columns(3)
        for i, (model_name, signal) in enumerate(prediction['individual_signals'].items()):
            with [col1, col2, col3][i % 3]:
                st.metric(
                    model_name,
                    "IN" if signal == 1 else "OUT"
                )
    
    # Historical analysis section
    st.header("Historical Analysis")
    
    history = fetch_history(ticker, history_days)
    
    if history:
        # Convert to DataFrame
        df = pd.DataFrame(history['predictions'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = history['performance_metrics']
        
        with col1:
            st.metric(
                "Strategy Return",
                f"{metrics['total_return']:.2%}"
            )
        
        with col2:
            st.metric(
                "Buy & Hold Return",
                f"{metrics['benchmark_return']:.2%}"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.2%}"
            )
        
        with col4:
            st.metric(
                "Number of Signals", 
                metrics['num_signals']
            )
        
        # Additional metrics if available
        if metrics.get('sharpe_ratio') is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.3f}"
                )
            with col2:
                if metrics.get('max_drawdown') is not None:
                    st.metric(
                        "Max Drawdown",
                        f"{metrics['max_drawdown']:.2%}"
                    )
            with col3:
                if metrics.get('volatility') is not None:
                    st.metric(
                        "Volatility",
                        f"{metrics['volatility']:.2%}"
                    )
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["Price & Signals", "Returns", "Data Table"])
        
        with tab1:
            st.plotly_chart(create_signal_chart(df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_returns_chart(df), use_container_width=True)
        
        with tab3:
            st.subheader("Historical Data")
            
            # Add download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Display table
            st.dataframe(
                df[['date', 'close', 'signal', 'return', 'strategy_return']].round(4),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Bitcoin Fall Prediction Dashboard**")
        st.caption("Powered by Generalized Fall Predictor Model")
    
    with col2:
        st.markdown("**API Status**")
        if api_healthy:
            st.caption("🟢 Connected")
        else:
            st.caption("🔴 Disconnected")
    
    with col3:
        st.markdown("**Model Info**")
        if model_status and model_status.get('loaded'):
            # Re-check demo status for footer
            is_demo_footer = (model_status.get('model_path', '').find('demo') != -1 or 
                             model_status.get('features_count', 0) == 30 and 
                             model_status.get('training_date', '').startswith('202'))
            model_type = "Demo" if is_demo_footer else "Production"
            st.caption(f"📊 {model_type} Model")
        else:
            st.caption("❌ No Model")


if __name__ == "__main__":
    main()