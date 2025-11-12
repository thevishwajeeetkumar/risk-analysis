"""
Database connection and session management for Neon PostgreSQL.

Uses asyncpg for async operations with the existing Neon database.
"""

import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Convert postgresql:// to postgresql+asyncpg:// and handle SSL parameters
def convert_to_asyncpg_url(url: str) -> str:
    """
    Convert PostgreSQL URL to asyncpg-compatible format.
    Removes sslmode and channel_binding (asyncpg handles SSL differently).
    """
    # Parse the URL
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Remove sslmode and channel_binding (asyncpg doesn't use these)
    # SSL is handled automatically by asyncpg when connecting to Neon
    if 'sslmode' in query_params:
        del query_params['sslmode']
    if 'channel_binding' in query_params:
        del query_params['channel_binding']
    
    # Reconstruct query string
    new_query = urlencode(query_params, doseq=True) if query_params else ''
    
    # Reconstruct URL
    new_parsed = parsed._replace(query=new_query)
    new_url = urlunparse(new_parsed)
    
    # Convert postgresql:// to postgresql+asyncpg://
    async_url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', new_url)
    
    return async_url

ASYNC_DATABASE_URL = convert_to_asyncpg_url(DATABASE_URL)

# For asyncpg with Neon, we need to ensure SSL is enabled
# We'll pass ssl=True via connect_args

# Create async engine with Neon-optimized settings
# For Neon, SSL is required - asyncpg handles this automatically
# Echo mode: Set DB_ECHO=true in development, false/unset in production
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=DB_ECHO,  # Controlled by environment variable
    pool_pre_ping=True,  # Important for Neon connection pooling - verifies connections before use
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,  # Wait up to 30 seconds for a connection from the pool
    connect_args={
        "ssl": True,  # Enable SSL for Neon connections (asyncpg uses True/False)
        "command_timeout": 10,  # Timeout for database commands (10 seconds)
    }
)

if DB_ECHO:
    print("[WARNING] Database echo mode is ENABLED (set DB_ECHO=false in production)")

# Note: command_timeout is set in connect_args above (10 seconds)
# This prevents long-running queries from blocking connections
# For statement-level timeouts, we can set them per-query if needed

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session for use in FastAPI endpoints
    """
    async with async_session_factory() as session:
        try:
            yield session
            # Only commit if no exception occurred
            await session.commit()
        except Exception as e:
            # Rollback on any exception
            try:
                await session.rollback()
                print(f"[ERROR] Database transaction rolled back: {str(e)}")
            except Exception as rollback_error:
                # If rollback itself fails, log but don't raise (session may already be closed)
                print(f"[WARNING] Error during rollback (may be expected): {str(rollback_error)}")
            raise
        finally:
            await session.close()


async def test_connection():
    """
    Test connection to Neon database.
    
    Returns:
        bool: True if connection successful, raises exception otherwise
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT current_database(), current_user"))
            db_name, user = result.fetchone()
            print(f"[SUCCESS] Connected to Neon database: {db_name} as {user}")
            
            # Test that our tables exist
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'loans', 'ecl_segment_calculation', 'permissions')
                ORDER BY table_name
            """)
            result = await conn.execute(tables_query)
            tables = [row[0] for row in result.fetchall()]
            
            if len(tables) == 4:
                print(f"[SUCCESS] All required tables found: {', '.join(tables)}")
            else:
                print(f"[WARNING] Expected 4 tables, found {len(tables)}: {', '.join(tables)}")
            
            return True
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise


async def close_db():
    """Close database connections."""
    await engine.dispose()


# Initialize models when this module is imported
from db.models import Base

# Note: We don't create tables here since they already exist in Neon
# Base.metadata would normally be used for table creation

