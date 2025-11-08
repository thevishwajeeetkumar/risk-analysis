"""Test file upload with authentication"""
import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("Testing File Upload Pipeline")
print("=" * 60)

# Step 1: Register/Login
print("\n1. Logging in...")
login_response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "test_user", "password": "TestPassword123"}
)

if login_response.status_code == 200:
    token = login_response.json()["access_token"]
    print(f"[SUCCESS] Logged in, token: {token[:20]}...")
else:
    print(f"[ERROR] Login failed: {login_response.status_code}")
    print(f"Response: {login_response.text}")
    sys.exit(1)

# Step 2: Upload file
print("\n2. Uploading CSV file...")
csv_file = Path("data/uploads/loan_data.csv")

if not csv_file.exists():
    print(f"[ERROR] Test CSV not found at {csv_file}")
    print("[INFO] Looking for any CSV file in data/uploads...")
    upload_dir = Path("data/uploads")
    csv_files = list(upload_dir.glob("*.csv"))
    if csv_files:
        csv_file = csv_files[0]
        print(f"[INFO] Using: {csv_file}")
    else:
        print("[ERROR] No CSV files found in data/uploads/")
        sys.exit(1)

with open(csv_file, 'rb') as f:
    files = {'file': (csv_file.name, f, 'text/csv')}
    headers = {'Authorization': f'Bearer {token}'}
    
    print(f"[INFO] Uploading {csv_file.name}...")
    response = requests.post(
        f"{BASE_URL}/api/upload",
        files=files,
        headers=headers
    )

print(f"\nResponse Status: {response.status_code}")
print(f"Response Content:")
print(json.dumps(response.json(), indent=2))

if response.status_code == 200 and response.json().get("status") == "success":
    print("\n[SUCCESS] Upload completed successfully!")
    stats = response.json().get("statistics", {})
    print(f"  Loans stored: {response.json().get('loan_count', 0)}")
    print(f"  Segments: {len(response.json().get('segments', []))}")
    print(f"  Total ECL: ${stats.get('total_ecl', 0):,.2f}")
else:
    print("\n[ERROR] Upload failed!")
    print(f"  Message: {response.json().get('message', 'Unknown error')}")

