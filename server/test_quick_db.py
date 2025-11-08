"""Quick database test"""
import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Load environment variables
load_dotenv()

async def quick_test():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("[ERROR] DATABASE_URL not found in environment")
        return
    
    print(f"[INFO] Database URL: {DATABASE_URL[:30]}...")
    
    # Convert to async URL and remove SSL params
    import re
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    parsed = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed.query)
    
    # Remove SSL params that asyncpg doesn't understand
    query_params.pop('sslmode', None)
    query_params.pop('channel_binding', None)
    
    new_query = urlencode(query_params, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    new_url = urlunparse(new_parsed)
    
    # Convert to asyncpg
    async_url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', new_url)
    
    print("[INFO] Connecting to database...")
    
    try:
        engine = create_async_engine(async_url, connect_args={"ssl": True})
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("[SUCCESS] Database connection successful!")
            
            # Check tables
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            print(f"\n[INFO] Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
                
        await engine.dispose()
        
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
