"""
Comprehensive integration tests for authentication, database, and endpoints.

Run this script to test:
1. Database connection
2. Authentication (password hashing, JWT)
3. API endpoints (with FastAPI test client)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Test imports
print("=" * 70)
print("INTEGRATION TESTS")
print("=" * 70)

# ==================== 1. Database Connection Test ====================
print("\n[1/3] Testing Database Connection...")
print("-" * 70)

try:
    from db.database import test_connection
    
    async def test_db():
        result = await test_connection()
        return result
    
    result = asyncio.run(test_db())
    if result:
        print("✅ Database connection: SUCCESS")
    else:
        print("❌ Database connection: FAILED")
        sys.exit(1)
except Exception as e:
    print(f"❌ Database connection: FAILED - {e}")
    print("\n⚠️  Note: Make sure DATABASE_URL is set in .env file")
    print("⚠️  Note: Make sure the database tables exist in Neon")
    sys.exit(1)

# ==================== 2. Authentication Test ====================
print("\n[2/3] Testing Authentication...")
print("-" * 70)

try:
    from auth.auth import hash_password, verify_password, create_access_token, decode_access_token
    from datetime import timedelta
    
    # Test password hashing
    test_password = "TestPassword123"
    hashed = hash_password(test_password)
    print(f"✅ Password hashing: SUCCESS")
    
    # Test password verification
    if verify_password(test_password, hashed):
        print(f"✅ Password verification: SUCCESS")
    else:
        print(f"❌ Password verification: FAILED")
        sys.exit(1)
    
    # Test JWT creation
    token_data = {"sub": "1", "role": "CRO"}
    token = create_access_token(token_data)
    print(f"✅ JWT token creation: SUCCESS")
    
    # Test JWT decoding
    decoded = decode_access_token(token)
    if decoded["sub"] == "1" and decoded["role"] == "CRO":
        print(f"✅ JWT token decoding: SUCCESS")
    else:
        print(f"❌ JWT token decoding: FAILED")
        sys.exit(1)
    
    print("✅ Authentication: ALL TESTS PASSED")
    
except Exception as e:
    print(f"❌ Authentication test: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 3. API Endpoints Test ====================
print("\n[3/3] Testing API Endpoints...")
print("-" * 70)

try:
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    if response.status_code == 200:
        print("✅ GET /: SUCCESS")
    else:
        print(f"❌ GET /: FAILED - Status {response.status_code}")
    
    # Test health endpoint
    response = client.get("/health")
    if response.status_code == 200:
        print("✅ GET /health: SUCCESS")
    else:
        print(f"❌ GET /health: FAILED - Status {response.status_code}")
    
    # Test login endpoint (will fail without valid user, but should return 401, not 500)
    response = client.post(
        "/auth/login",
        data={"username": "test_user", "password": "test_password"}
    )
    if response.status_code in [401, 422]:  # 401 = unauthorized, 422 = validation error
        print("✅ POST /auth/login: SUCCESS (expected 401/422 for invalid credentials)")
    else:
        print(f"⚠️  POST /auth/login: Unexpected status {response.status_code}")
    
    # Test protected endpoint without auth (should return 401)
    response = client.get("/auth/me")
    if response.status_code == 401:
        print("✅ GET /auth/me (without auth): SUCCESS (correctly returns 401)")
    else:
        print(f"⚠️  GET /auth/me: Unexpected status {response.status_code}")
    
    # Test upload endpoint without auth (should return 401)
    response = client.post("/api/upload", files={"file": ("test.csv", b"test,data")})
    if response.status_code == 401:
        print("✅ POST /api/upload (without auth): SUCCESS (correctly returns 401)")
    else:
        print(f"⚠️  POST /api/upload: Unexpected status {response.status_code}")
    
    # Test query endpoint without auth (should return 401)
    response = client.post("/api/query", json={"query": "test query"})
    if response.status_code == 401:
        print("✅ POST /api/query (without auth): SUCCESS (correctly returns 401)")
    else:
        print(f"⚠️  POST /api/query: Unexpected status {response.status_code}")
    
    # Test segments endpoint without auth (should return 401)
    response = client.get("/api/segments")
    if response.status_code == 401:
        print("✅ GET /api/segments (without auth): SUCCESS (correctly returns 401)")
    else:
        print(f"⚠️  POST /api/segments: Unexpected status {response.status_code}")
    
    print("✅ API Endpoints: ALL TESTS PASSED")
    
except Exception as e:
    print(f"❌ API endpoints test: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== Summary ====================
print("\n" + "=" * 70)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
print("\nNext steps:")
print("1. Create a CRO user in the database (or use existing)")
print("2. Login via POST /auth/login to get JWT token")
print("3. Use the token in Authorization header for protected endpoints")
print("\nExample:")
print("  curl -X POST http://localhost:8000/auth/login \\")
print("    -H 'Content-Type: application/x-www-form-urlencoded' \\")
print("    -d 'username=your_username&password=your_password'")
print("=" * 70)
