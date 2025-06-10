#!/usr/bin/env python3
"""
Test script for the simplified Bitcoin Fall Prediction API
"""
import requests
import json
import time
import sys

def test_simple_api(base_url="http://localhost:8000"):
    """Test all endpoints for the simplified API"""
    
    print(f"Testing Simplified Bitcoin Fall Prediction API")
    print(f"API URL: {base_url}")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {response.status_code}")
            print(f"   API: {data['name']} v{data['version']}")
            print(f"   Description: {data['description']}")
            tests_passed += 1
        else:
            print(f"❌ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {response.status_code}")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            tests_passed += 1
        else:
            print(f"❌ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 3: Model status
    try:
        response = requests.get(f"{base_url}/model/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model status: {response.status_code}")
            print(f"   Loaded: {data['loaded']}")
            if data['loaded']:
                print(f"   Ticker: {data['ticker']}")
                print(f"   Features: {data['features_count']}")
                print(f"   Mode: {data.get('training_date', 'Unknown')}")
            tests_passed += 1
        else:
            print(f"❌ Model status: {response.status_code}")
    except Exception as e:
        print(f"❌ Model status failed: {e}")
    
    # Test 4: Current prediction
    try:
        response = requests.get(f"{base_url}/predict/today?ticker=BTC-USD", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Current prediction: {response.status_code}")
            print(f"   Date: {data['date']}")
            print(f"   Signal (with uptrend): {data['signal_with_uptrend']}")
            print(f"   Signal (no filter): {data['signal_without_uptrend']:.3f}")
            print(f"   Ensemble score: {data['ensemble_score']:.3f}")
            print(f"   Uptrend active: {data['uptrend_active']}")
            print(f"   Individual signals: {data['individual_signals']}")
            tests_passed += 1
        else:
            print(f"❌ Current prediction: {response.status_code}")
            print(f"   Error: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Current prediction failed: {e}")
    
    # Test 5: Historical predictions
    try:
        response = requests.get(f"{base_url}/predict/history?ticker=BTC-USD&days=7", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Historical predictions: {response.status_code}")
            print(f"   Period: {data['start_date']} to {data['end_date']}")
            print(f"   Data points: {len(data['predictions'])}")
            
            metrics = data['performance_metrics']
            print(f"   Strategy return: {metrics['total_return']:.3f}")
            print(f"   Benchmark return: {metrics['benchmark_return']:.3f}")
            print(f"   Signals: {metrics['num_signals']}")
            print(f"   Win rate: {metrics['win_rate']:.3f}")
            
            if metrics.get('sharpe_ratio'):
                print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            
            tests_passed += 1
        else:
            print(f"❌ Historical predictions: {response.status_code}")
            print(f"   Error: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Historical predictions failed: {e}")
    
    # Test 6: Training jobs (should show disabled)
    try:
        response = requests.get(f"{base_url}/training/jobs", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training jobs: {response.status_code}")
            print(f"   Total jobs: {data['total']}")
            if 'message' in data:
                print(f"   Status: {data['message']}")
            tests_passed += 1
        else:
            print(f"❌ Training jobs: {response.status_code}")
    except Exception as e:
        print(f"❌ Training jobs failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"API Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working perfectly.")
        print("\n📊 Ready for:")
        print(f"   • Streamlit Dashboard: streamlit run streamlit_app/app.py")
        print(f"   • API Documentation: {base_url}/docs")
        print(f"   • Integration: Use the API endpoints in your applications")
    elif tests_passed >= 4:
        print("⚠️ Most tests passed. API should work with minor issues.")
    else:
        print("🚨 Several tests failed. Check API configuration.")
    
    return tests_passed == total_tests


def test_different_tickers(base_url="http://localhost:8000"):
    """Test API with different tickers"""
    print(f"\n🔄 Testing different tickers...")
    
    tickers = ["BTC-USD", "ETH-USD", "AAPL"]
    
    for ticker in tickers:
        try:
            response = requests.get(
                f"{base_url}/predict/today?ticker={ticker}", 
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {ticker}: Signal = {data['signal_with_uptrend']}")
            else:
                print(f"❌ {ticker}: {response.status_code}")
        except Exception as e:
            print(f"❌ {ticker}: {str(e)[:50]}")


def test_api_performance(base_url="http://localhost:8000"):
    """Test API response times"""
    print(f"\n⚡ Testing API performance...")
    
    endpoints = [
        "/health",
        "/model/status", 
        "/predict/today?ticker=BTC-USD"
    ]
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ {endpoint}: {duration:.3f}s")
            else:
                print(f"❌ {endpoint}: {response.status_code} ({duration:.3f}s)")
        except Exception as e:
            print(f"❌ {endpoint}: {str(e)[:50]}")


if __name__ == "__main__":
    # Allow custom API URL
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Main API tests
    success = test_simple_api(api_url)
    
    if success:
        # Additional tests if main tests pass
        test_different_tickers(api_url)
        test_api_performance(api_url)
        
        print(f"\n🚀 API is ready for production!")
        print(f"   • Start Streamlit: cd streamlit_app && streamlit run app.py")
        print(f"   • API Docs: {api_url}/docs")
        print(f"   • Example: curl \"{api_url}/predict/today?ticker=BTC-USD\"")
    else:
        print(f"\n🔧 Fix the issues above and try again.")
        sys.exit(1)