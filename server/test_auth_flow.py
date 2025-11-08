"""
Test script to verify registration and login flow
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db.database import get_db, test_connection
from db import crud
from auth.auth import hash_password, verify_password, authenticate_user

async def test_registration_and_login():
    """Test the complete registration and login flow"""
    print("\n" + "="*70)
    print("TESTING REGISTRATION AND LOGIN FLOW")
    print("="*70)
    
    # Test database connection
    print("\n1. Testing database connection...")
    try:
        await test_connection()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # Test user creation
    print("\n2. Testing user creation...")
    import time
    test_username = f"test_user_{int(time.time())}"
    test_email = f"{test_username}@test.com"
    test_password = "testpass123"
    test_role = "Analyst"
    
    try:
        async for db in get_db():
            # Check if user exists
            existing = await crud.get_user_by_username(db, test_username)
            if existing:
                print(f"⚠️  Test user already exists: {test_username}")
                # Try to delete it first
                await db.delete(existing)
                await db.commit()
            
            # Create user
            hashed = hash_password(test_password)
            print(f"   Password hash: {hashed[:30]}...")
            
            user = await crud.create_user(
                db,
                username=test_username,
                password_hash=hashed,
                email=test_email,
                role=test_role
            )
            print(f"✅ User created: {user.username} (ID: {user.user_id})")
            break
    except Exception as e:
        print(f"❌ User creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test password verification
    print("\n3. Testing password verification...")
    try:
        async for db in get_db():
            user = await crud.get_user_by_username(db, test_username)
            if not user:
                print("❌ User not found after creation")
                return False
            
            is_valid = verify_password(test_password, user.password_hash)
            if is_valid:
                print("✅ Password verification successful")
            else:
                print("❌ Password verification failed")
                return False
            break
    except Exception as e:
        print(f"❌ Password verification test failed: {e}")
        return False
    
    # Test authentication
    print("\n4. Testing authentication...")
    try:
        async for db in get_db():
            user = await authenticate_user(db, test_username, test_password)
            if user:
                print(f"✅ Authentication successful: {user.username}")
            else:
                print("❌ Authentication failed")
                return False
            break
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    result = asyncio.run(test_registration_and_login())
    sys.exit(0 if result else 1)

