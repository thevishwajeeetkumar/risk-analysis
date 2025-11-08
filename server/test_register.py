"""
Test script to verify user registration endpoint.
Run this to test if registration is working properly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db.database import get_db
from db import crud
from auth.auth import hash_password, verify_password

async def test_registration():
    """Test user registration flow."""
    print("Testing user registration...")
    
    test_username = "test_user_" + str(asyncio.get_event_loop().time())[:10].replace('.', '')
    test_email = f"{test_username}@test.com"
    test_password = "testpass123"
    
    async for db in get_db():
        try:
            # Check if user exists
            existing = await crud.get_user_by_username(db, test_username)
            if existing:
                print(f"⚠️  Test user {test_username} already exists, skipping...")
                return
            
            # Create user
            print(f"Creating user: {test_username}")
            hashed_password = hash_password(test_password)
            user = await crud.create_user(
                db,
                username=test_username,
                password_hash=hashed_password,
                email=test_email,
                role="Analyst"
            )
            
            print(f"✅ User created successfully!")
            print(f"   User ID: {user.user_id}")
            print(f"   Username: {user.username}")
            print(f"   Email: {user.email}")
            print(f"   Role: {user.role}")
            
            # Verify password
            if verify_password(test_password, user.password_hash):
                print("✅ Password verification successful")
            else:
                print("❌ Password verification failed")
            
            # Clean up - delete test user
            await db.delete(user)
            await db.commit()
            print("✅ Test user cleaned up")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        break

if __name__ == "__main__":
    asyncio.run(test_registration())

