"""
Quick test script for API endpoints
Run this to verify all endpoints are working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None, files=None):
    """Test an endpoint and print results"""
    url = f"{BASE_URL}{path}"
    print(f"\n{'='*60}")
    print(f"Testing: {method} {path}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text[:500])
        
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ECL API ENDPOINT TESTER")
    print("="*60)
    
    # Test 1: Root endpoint
    test_endpoint("GET", "/")
    
    # Test 2: Health check
    test_endpoint("GET", "/health")
    
    # Test 3: Segments endpoint
    test_endpoint("GET", "/api/segments")
    
    # Test 4: Query endpoint (will fail without data, but tests endpoint exists)
    test_endpoint("POST", "/api/query", data={"query": "test query"})
    
    # Test 5: Docs endpoint
    print(f"\n{'='*60}")
    print("Testing: GET /docs")
    print(f"{'='*60}")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"Status Code: {response.status_code}")
        print("✓ Swagger UI is accessible")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print("\nTo test file upload, use:")
    print(f"  curl -X POST \"{BASE_URL}/api/upload\" -F \"file=@data/uploads/loan_data.csv\"")
    print("\nOr visit in browser:")
    print(f"  {BASE_URL}/docs")

