"""Database schema checking utilities."""

from typing import Dict, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


_COLUMN_CACHE: Dict[Tuple[str, str, str], bool] = {}


LOANS_REQUIRED_COLUMNS = [
    {
        "name": "user_id",
        "type": "INTEGER",
        "nullable": True,
        "references": '"public"."users"("user_id")',
    },
    {"name": "loan_amount", "type": "DOUBLE PRECISION", "nullable": True},
    {"name": "loan_intent", "type": "VARCHAR(64)", "nullable": True},
    {"name": "loan_int_rate", "type": "DOUBLE PRECISION", "nullable": True},
    {"name": "loan_percent_income", "type": "DOUBLE PRECISION", "nullable": True},
    {"name": "credit_score", "type": "INTEGER", "nullable": True},
    {"name": "person_income", "type": "DOUBLE PRECISION", "nullable": True},
    {"name": "person_age", "type": "INTEGER", "nullable": True},
    {"name": "person_gender", "type": "VARCHAR(20)", "nullable": True},
    {"name": "person_education", "type": "VARCHAR(50)", "nullable": True},
    {"name": "person_emp_exp", "type": "INTEGER", "nullable": True},
    {"name": "home_ownership", "type": "VARCHAR(20)", "nullable": True},
    {"name": "cb_person_cred_hist_length", "type": "INTEGER", "nullable": True},
    {
        "name": "previous_loan_defaults_on_file",
        "type": "BOOLEAN",
        "nullable": True,
        "default": "FALSE",
    },
    {"name": "loan_status", "type": "INTEGER", "nullable": True},
    {
        "name": "created_at",
        "type": "TIMESTAMP WITH TIME ZONE",
        "nullable": True,
        "default": "CURRENT_TIMESTAMP",
    },
]


async def column_exists(
    db: AsyncSession,
    table_name: str,
    column_name: str,
    schema: str = "public"
) -> bool:
    """Check if a column exists in a database table."""

    cache_key = (schema, table_name, column_name)
    if cache_key in _COLUMN_CACHE:
        return _COLUMN_CACHE[cache_key]

    try:
        query = text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
              AND column_name = :column_name
            LIMIT 1
            """
        )
        result = await db.execute(
            query,
            {"schema": schema, "table_name": table_name, "column_name": column_name},
        )
        exists = result.fetchone() is not None
        _COLUMN_CACHE[cache_key] = exists
        return exists
    except Exception as e:
        print(f"[WARNING] Error checking column existence: {str(e)}")
        return False


async def get_table_columns(
    db: AsyncSession,
    table_name: str,
    schema: str = "public"
) -> list:
    """
    Get all column names for a table.
    
    Args:
        db: Database session
        table_name: Name of the table
        
    Returns:
        List of column names
    """
    try:
        query = text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
            ORDER BY ordinal_position
            """
        )
        result = await db.execute(
            query,
            {"schema": schema, "table_name": table_name},
        )
        return [row[0] for row in result.fetchall()]
    except Exception as e:
        print(f"[WARNING] Error getting table columns: {str(e)}")
        return []


async def add_column_if_missing(
    db: AsyncSession,
    table_name: str,
    column_name: str,
    column_type: str,
    schema: str = "public",
    nullable: bool = True,
    default_value: str = None,
    references: str | None = None
) -> bool:
    """
    Add a column to a table if it doesn't exist.
    
    Args:
        db: Database session
        table_name: Name of the table
        column_name: Name of the column to add
        column_type: SQL type (e.g., 'INTEGER', 'VARCHAR(255)')
        nullable: Whether column can be NULL
        default_value: Default value for the column
        
    Returns:
        True if column was added or already exists, False on error
    """
    try:
        # Check if column exists
        exists = await column_exists(db, table_name, column_name, schema=schema)
        if exists:
            print(f"[SUCCESS] Column {table_name}.{column_name} already exists")
            return True
        
        # Build ALTER TABLE statement
        nullable_clause = "" if nullable else " NOT NULL"
        default_clause = f" DEFAULT {default_value}" if default_value else ""

        reference_clause = f" REFERENCES {references}" if references else ""

        table_identifier = (
            f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
        )
        column_identifier = f'"{column_name}"'

        alter_sql = (
            f"ALTER TABLE {table_identifier} "
            f"ADD COLUMN {column_identifier} {column_type}{reference_clause}{nullable_clause}{default_clause}"
        )

        await db.execute(text(alter_sql))
        await db.commit()
        print(f"[SUCCESS] Added column {table_name}.{column_name}")
        _COLUMN_CACHE[(schema, table_name, column_name)] = True
        return True
    except Exception as e:
        print(f"[ERROR] Error adding column {table_name}.{column_name}: {str(e)}")
        await db.rollback()
        return False


async def drop_constraint_if_exists(
    db: AsyncSession,
    table_name: str,
    constraint_name: str,
    schema: str = "public"
) -> bool:
    """Drop a constraint if it exists."""

    try:
        # Check if constraint exists
        check_query = text(
            """
            SELECT 1
            FROM information_schema.table_constraints
            WHERE constraint_schema = :schema
              AND table_name = :table_name
              AND constraint_name = :constraint_name
            LIMIT 1
            """
        )
        result = await db.execute(
            check_query,
            {"schema": schema, "table_name": table_name, "constraint_name": constraint_name},
        )
        exists = result.fetchone() is not None

        if not exists:
            return True

        # Drop the constraint
        table_identifier = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
        drop_sql = f'ALTER TABLE {table_identifier} DROP CONSTRAINT IF EXISTS "{constraint_name}"'

        await db.execute(text(drop_sql))
        await db.commit()
        print(f"[SUCCESS] Dropped constraint {constraint_name} from {table_name}")
        return True
    except Exception as e:
        print(f"[WARNING] Error dropping constraint {constraint_name}: {str(e)}")
        return False


async def drop_all_check_constraints(
    db: AsyncSession,
    table_name: str,
    schema: str = "public"
) -> bool:
    """Drop all CHECK constraints from a table."""

    try:
        fetch_sql = text(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema = :schema
              AND table_name = :table_name
              AND constraint_type = 'CHECK'
            """
        )

        result = await db.execute(fetch_sql, {"schema": schema, "table_name": table_name})
        constraints = [row[0] for row in result.fetchall()]

        if not constraints:
            return True

        for constraint_name in constraints:
            drop_sql = text(
                f'ALTER TABLE "{schema}"."{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"'
            )
            await db.execute(drop_sql)

        await db.commit()
        print(f"[INFO] Dropped {len(constraints)} CHECK constraints from {table_name}")
        return True
    except Exception as e:
        await db.rollback()
        print(f"[ERROR] Failed to drop CHECK constraints from {table_name}: {str(e)}")
        return False


async def ensure_loans_schema(db: AsyncSession) -> bool:
    """Ensure the loans table contains all required columns."""

    try:
        # Drop legacy CHECK constraints that might conflict
        await drop_constraint_if_exists(db, "loans", "loans_person_gender_check")
        await drop_constraint_if_exists(db, "loans", "loans_person_education_check")
        await drop_constraint_if_exists(db, "loans", "loans_home_ownership_check")
        await drop_constraint_if_exists(db, "loans", "loans_loan_intent_check")
        # Add more as needed - comprehensive removal recommended

        all_ok = True
        for column in LOANS_REQUIRED_COLUMNS:
            added = await add_column_if_missing(
                db,
                table_name="loans",
                column_name=column["name"],
                column_type=column["type"],
                schema="public",
                nullable=column.get("nullable", True),
                default_value=column.get("default"),
                references=column.get("references"),
            )
            all_ok = all_ok and added
        if all_ok:
            print("[SUCCESS] loans table schema verified")
        else:
            print("[WARNING] loans table schema partially verified; check logs for failures")
        return all_ok
    except Exception as e:
        print(f"[ERROR] Failed to ensure loans table schema: {str(e)}")
        return False


async def ensure_ecl_schema(db: AsyncSession) -> bool:
    """
    Ensure ECL segment calculation table has no restrictive CHECK constraints.
    
    Drops all CHECK constraints that might restrict segment_name or other values.
    This allows dynamic segment types like 'person_home_ownership' and 'age_group'.
    """
    try:
        # Drop all CHECK constraints (allows new segment types)
        await drop_all_check_constraints(db, "ecl_segment_calculation")

        # Ensure loan_id can be NULL for aggregate calculations
        check_nullable_sql = text(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
              AND column_name = 'loan_id'
            LIMIT 1
            """
        )
        result = await db.execute(check_nullable_sql, {"schema": "public", "table_name": "ecl_segment_calculation"})
        row = result.fetchone()

        if row and row[0] == "NO":
            alter_sql = text(
                'ALTER TABLE "public"."ecl_segment_calculation" ALTER COLUMN "loan_id" DROP NOT NULL'
            )
            await db.execute(alter_sql)
            await db.commit()
            print("[INFO] Updated ecl_segment_calculation.loan_id to allow NULL")

        print("[SUCCESS] ecl_segment_calculation table schema verified")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to ensure ECL table schema: {str(e)}")
        return False

