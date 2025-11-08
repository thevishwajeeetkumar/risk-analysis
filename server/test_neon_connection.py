"""
Test script to verify Neon PostgreSQL connection.

Run this script to ensure the database connection is working properly.
"""

import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment or direct specification
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://neondb_owner:npg_cSvNDbl72dIF@ep-icy-credit-ahvkz02e-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')

# Convert postgresql:// to postgresql+asyncpg:// and handle SSL parameters
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def convert_to_asyncpg_url(url: str) -> str:
    """Convert PostgreSQL URL to asyncpg-compatible format."""
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Remove sslmode and channel_binding (asyncpg doesn't use these)
    if 'sslmode' in query_params:
        del query_params['sslmode']
    if 'channel_binding' in query_params:
        del query_params['channel_binding']
    
    new_query = urlencode(query_params, doseq=True) if query_params else ''
    new_parsed = parsed._replace(query=new_query)
    new_url = urlunparse(new_parsed)
    async_url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', new_url)
    return async_url

ASYNC_DATABASE_URL = convert_to_asyncpg_url(DATABASE_URL)


async def test_connection():
    """Test the connection to Neon database."""
    print("Testing Neon PostgreSQL connection...")
    print(f"Database URL (masked): {DATABASE_URL.split('@')[0]}@***")
    
    try:
        # Create async engine with SSL enabled for Neon
        engine = create_async_engine(
            ASYNC_DATABASE_URL,
            echo=True,
            connect_args={"ssl": True}  # Enable SSL for Neon
        )
        
        async with engine.connect() as conn:
            # Test basic connectivity
            result = await conn.execute(text("SELECT current_database(), current_user, version()"))
            db_name, user, version = result.fetchone()
            
            print("\n✅ Connection successful!")
            print(f"Database: {db_name}")
            print(f"User: {user}")
            print(f"PostgreSQL Version: {version.split(',')[0]}")
            
            # Check for required tables
            print("\nChecking for required tables...")
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'loans', 'ecl_segment_calculation', 'permissions')
                ORDER BY table_name
            """)
            
            result = await conn.execute(tables_query)
            tables = [row[0] for row in result.fetchall()]
            
            print(f"\nFound {len(tables)} tables:")
            for table in tables:
                print(f"  ✓ {table}")
            
            missing_tables = []
            required_tables = ['users', 'loans', 'ecl_segment_calculation', 'permissions']
            for req_table in required_tables:
                if req_table not in tables:
                    missing_tables.append(req_table)
            
            if missing_tables:
                print(f"\n⚠️  Warning: Missing tables: {', '.join(missing_tables)}")
                print("These tables need to be created in your Neon database.")
            else:
                print("\n✅ All required tables exist!")
            
        await engine.dispose()
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. DATABASE_URL is correct in your .env file")
        print("2. The database is accessible")
        print("3. The credentials are valid")
        raise


if __name__ == "__main__":
    asyncio.run(test_connection())
