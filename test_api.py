"""
Test script for the Bitcoin Fall Prediction API
"""
import requests
import json
import time

def test_api(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    
    print(f"Testing API at {base_url}")
    print("=" * 50)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        root_data = response.json()
        print(f"   API: {root_data['name']} v{root_data['version']}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        health_data = response.json()
        print(f"   Status: {health_data['status']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test model status
    try:
        response = requests.get(f"{base_url}/model/status")
        print(f"✅ Model status: {response.status_code}")
        status = response.json()
        print(f"   Model loaded: {status['loaded']}")
        if status['loaded']:
            print(f"   Ticker: {status['ticker']}")
            print(f"   Training date: {status['training_date']}")
            print(f"   Features count: {status['features_count']}")
    except Exception as e:
        print(f"❌ Model status failed: {e}")
    
    # Test current prediction
    try:
        response = requests.get(f"{base_url}/predict/today?ticker=BTC-USD")
        print(f"✅ Current prediction: {response.status_code}")
        pred = response.json()
        print(f"   Date: {pred['date']}")
        print(f"   Signal (with uptrend): {pred['signal_with_uptrend']}")
        print(f"   Signal (no filter): {pred['signal_without_uptrend']:.3f}")
        print(f"   Ensemble score: {pred['ensemble_score']:.3f}")
        print(f"   Uptrend active: {pred['uptrend_active']}")
        print(f"   Individual signals: {pred['individual_signals']}")
    except Exception as e:
        print(f"❌ Current prediction failed: {e}")
    
    # Test historical predictions
    try:
        response = requests.get(f"{base_url}/predict/history?ticker=BTC-USD&days=7")
        print(f"✅ Historical predictions: {response.status_code}")
        hist = response.json()
        print(f"   Period: {hist['start_date']} to {hist['end_date']}")
        metrics = hist['performance_metrics']
        print(f"   Strategy return: {metrics['total_return']:.3f}")
        print(f"   Benchmark return: {metrics['benchmark_return']:.3f}")
        print(f"   Number of signals: {metrics['num_signals']}")
        print(f"   Win rate: {metrics['win_rate']:.3f}")
        if metrics['sharpe_ratio']:
            print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"❌ Historical predictions failed: {e}")
    
    # Test training jobs list (simplified - should return empty)
    try:
        response = requests.get(f"{base_url}/training/jobs")
        print(f"✅ Training jobs list: {response.status_code}")
        jobs_data = response.json()
        print(f"   Total jobs: {jobs_data['total']}")
        print(f"   Message: {jobs_data.get('message', 'N/A')}")
    except Exception as e:
        print(f"❌ Training jobs list failed: {e}")
    
    # Test invalid ticker (should return proper error)
    try:
        response = requests.get(f"{base_url}/predict/today?ticker=INVALID")
        print(f"✅ Invalid ticker test: {response.status_code}")
        if response.status_code == 404:
            error_data = response.json()
            print(f"   Error message: {error_data['message']}")
        else:
            print(f"   Unexpected status code for invalid ticker")
    except Exception as e:
        print(f"❌ Invalid ticker test failed: {e}")
    
    print("\n" + "=" * 50)
    print("API test completed!")

if __name__ == "__main__":
    import sys
    
    # Allow custom URL as argument
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Test main API
    test_api(url)
    