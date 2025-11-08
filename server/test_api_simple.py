"""Simple API test with better error handling"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Test 1: Health check
print("1. Testing root endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)[:200]}...")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Try registration with verbose output
print("\n2. Testing registration...")
user_data = {
    "username": "testuser123",
    "email": "test123@example.com", 
    "password": "TestPassword123",
    "role": "Analyst"  # Add role field
}

try:
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=user_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code}")
    print(f"   Headers: {dict(response.headers)}")
    print(f"   Response: {response.text[:500]}")
    
    if response.status_code == 500:
        # Try to get more info
        print("\n   [DEBUG] Server returned 500 error")
        print("   This usually means:")
        print("   - Database schema mismatch")
        print("   - Missing required fields")
        print("   - Constraint violations")
        
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Check if user exists
print("\n3. Trying login with same credentials...")
login_data = {
    "username": "testuser123",
    "password": "TestPassword123"
}

try:
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data=login_data,  # Form data for OAuth2
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   [SUCCESS] Login worked - user exists!")
        token = response.json()["access_token"]
        print(f"   Token: {token[:20]}...")
    else:
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\nDone!")
