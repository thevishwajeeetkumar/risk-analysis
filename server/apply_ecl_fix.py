"""
Apply database fix for ecl_segment_calculation.loan_id constraint
"""

import asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from db.database import engine

load_dotenv()

async def apply_fix():
    print("=" * 60)
    print("Fixing ecl_segment_calculation.loan_id constraint")
    print("=" * 60)
    
    try:
        async with engine.connect() as conn:
            # Check current constraint
            print("\n[1/3] Checking current constraint...")
            check_sql = text("""
                SELECT column_name, is_nullable, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'ecl_segment_calculation' 
                AND column_name = 'loan_id'
            """)
            
            result = await conn.execute(check_sql)
            row = result.fetchone()
            
            if row:
                print(f"   Column: {row[0]}")
                print(f"   Type: {row[2]}")
                print(f"   Nullable: {row[1]}")
                
                if row[1] == 'NO':
                    print("\n[2/3] Applying fix...")
                    fix_sql = text("""
                        ALTER TABLE ecl_segment_calculation 
                        ALTER COLUMN loan_id DROP NOT NULL
                    """)
                    
                    await conn.execute(fix_sql)
                    await conn.commit()
                    
                    print("[SUCCESS] Constraint removed successfully!")
                    
                    # Verify
                    print("\n[3/3] Verifying fix...")
                    result = await conn.execute(check_sql)
                    row = result.fetchone()
                    print(f"   Column: {row[0]}")
                    print(f"   Nullable: {row[1]}")
                    
                    if row[1] == 'YES':
                        print("\n[SUCCESS] Fix verified! loan_id can now be NULL")
                    else:
                        print("\n[ERROR] Fix failed - still NOT NULL")
                else:
                    print("\n[INFO] Column already allows NULL - no fix needed")
            else:
                print("[ERROR] Column loan_id not found in ecl_segment_calculation table")
                
    except Exception as e:
        print(f"\n[ERROR] Failed to apply fix: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(apply_fix())

