"""
Apply ECL CHECK constraints fix to database
Removes all CHECK constraints from ecl_segment_calculation table
"""

import asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from db.database import engine

load_dotenv()

async def drop_all_ecl_check_constraints():
    print("=" * 60)
    print("Dropping CHECK constraints from ecl_segment_calculation")
    print("=" * 60)
    
    try:
        async with engine.connect() as conn:
            # Step 1: Find all CHECK constraints
            print("\n[1/3] Finding CHECK constraints...")
            find_constraints_sql = text("""
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_name = 'ecl_segment_calculation'
                AND constraint_type = 'CHECK'
                ORDER BY constraint_name
            """)
            
            result = await conn.execute(find_constraints_sql)
            constraints = [row[0] for row in result.fetchall()]
            
            if not constraints:
                print("[INFO] No CHECK constraints found on ecl_segment_calculation")
                return True
            
            print(f"[INFO] Found {len(constraints)} CHECK constraints:")
            for constraint in constraints:
                print(f"   - {constraint}")
            
            # Step 2: Drop each constraint
            print(f"\n[2/3] Dropping {len(constraints)} constraints...")
            for constraint_name in constraints:
                drop_sql = text(f"""
                    ALTER TABLE ecl_segment_calculation 
                    DROP CONSTRAINT IF EXISTS "{constraint_name}"
                """)
                await conn.execute(drop_sql)
                print(f"   [SUCCESS] Dropped {constraint_name}")
            
            await conn.commit()
            
            # Step 3: Verify all dropped
            print("\n[3/3] Verifying...")
            result = await conn.execute(find_constraints_sql)
            remaining = [row[0] for row in result.fetchall()]
            
            if not remaining:
                print("[SUCCESS] All CHECK constraints removed successfully!")
                return True
            else:
                print(f"[WARNING] {len(remaining)} constraints still remain:")
                for c in remaining:
                    print(f"   - {c}")
                return False
                
    except Exception as e:
        print(f"\n[ERROR] Failed to drop constraints: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await engine.dispose()

if __name__ == "__main__":
    success = asyncio.run(drop_all_ecl_check_constraints())
    exit(0 if success else 1)

