"""Simple database table check"""
import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Convert to async URL
async_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

async def check_tables():
    engine = create_async_engine(async_url, echo=True)
    
    try:
        async with engine.connect() as conn:
            # Check if users table exists
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'loans', 'ecl_segment_calculation', 'permissions', 'rag_documents')
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            print(f"\nFound {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
            
            # Check users table structure
            result = await conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'users'
                ORDER BY ordinal_position
            """))
            
            print("\nUsers table columns:")
            for col in result.fetchall():
                print(f"  - {col[0]}: {col[1]} (nullable: {col[2]})")
                
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_tables())
