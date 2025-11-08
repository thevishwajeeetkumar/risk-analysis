#!/usr/bin/env python3
"""
Quick test script to demonstrate API authentication flow
"""

import requests
import json

# Base URL
BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("Testing ECL Risk Analysis API Authentication")
print("=" * 60)

# Test user credentials
username = "test_user"
email = "test@example.com"
password = "TestPassword123"

# Step 1: Register a new user
print("\n1. Registering new user...")
register_url = f"{BASE_URL}/auth/register"
register_data = {
    "username": username,
    "email": email,
    "password": password
}

try:
    response = requests.post(register_url, json=register_data)
    if response.status_code == 201:
        print(f"[SUCCESS] User registered: {username}")
        print(f"Response: {response.json()}")
    elif response.status_code == 400:
        print(f"[INFO] User already exists: {username}")
    else:
        print(f"[ERROR] Registration failed: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] Request failed: {e}")
    exit(1)

# Step 2: Login to get token
print("\n2. Logging in to get JWT token...")
login_url = f"{BASE_URL}/auth/login"
login_data = {
    "username": username,
    "password": password
}

try:
    # For OAuth2 password flow, use form data
    response = requests.post(login_url, data=login_data)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        print(f"[SUCCESS] Login successful!")
        print(f"Token type: {token_data['token_type']}")
        print(f"Token (first 20 chars): {access_token[:20]}...")
    else:
        print(f"[ERROR] Login failed: {response.status_code}")
        print(f"Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"[ERROR] Request failed: {e}")
    exit(1)

# Step 3: Test authenticated endpoint
print("\n3. Testing authenticated endpoint (GET /auth/me)...")
headers = {
    "Authorization": f"Bearer {access_token}"
}

try:
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if response.status_code == 200:
        user_data = response.json()
        print(f"[SUCCESS] Authenticated as:")
        print(json.dumps(user_data, indent=2))
    else:
        print(f"[ERROR] Authentication failed: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] Request failed: {e}")

# Step 4: Test protected ECL endpoints
print("\n4. Testing protected endpoint (GET /api/segments)...")
try:
    response = requests.get(f"{BASE_URL}/api/segments", headers=headers)
    if response.status_code == 200:
        segments_data = response.json()
        print(f"[SUCCESS] Segments endpoint accessible")
        print(f"Response preview: {json.dumps(segments_data, indent=2)[:200]}...")
    else:
        print(f"[INFO] Segments endpoint returned: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
except Exception as e:
    print(f"[ERROR] Request failed: {e}")

print("\n" + "=" * 60)
print("Authentication test completed!")
print("=" * 60)

print("\n[TIP] To test file upload:")
print(f"curl -X POST '{BASE_URL}/api/upload' \\")
print(f"  -H 'Authorization: Bearer {access_token[:20]}...' \\")
print("  -F 'file=@your_loan_data.csv'")

print("\n[TIP] To test RAG query:")
print(f"curl -X POST '{BASE_URL}/api/query' \\")
print(f"  -H 'Authorization: Bearer {access_token[:20]}...' \\")
print("  -H 'Content-Type: application/json' \\")
print("  -d '{\"query\": \"What segments have the highest ECL?\"}'")
