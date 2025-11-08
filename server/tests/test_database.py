"""
Database operations and authentication tests.

Tests CRUD operations, authentication flow, and database connectivity.
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import engine, get_db, test_connection
from db import crud
from db.models import User, Loan, ECLSegmentCalculation, Permission
from auth.auth import hash_password, verify_password, create_access_token, decode_access_token


# Test fixtures
@pytest.fixture
async def db_session():
    """Get a test database session."""
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "username": "test_user",
        "password": "TestPassword123",
        "email": "test@example.com",
        "role": "Analyst"
    }


@pytest.fixture
def test_loan_data():
    """Sample loan data for testing."""
    return {
        "loan_amount": 10000.0,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 5.5,
        "loan_percent_income": 0.15,
        "credit_score": 700,
        "person_income": 60000.0,
        "person_age": 35,
        "person_gender": "male",
        "person_education": "Bachelor",
        "person_emp_exp": 10,
        "home_ownership": "RENT",
        "cb_person_cred_hist_length": 5,
        "previous_loan_defaults_on_file": "No",
        "loan_status": 1
    }


# Database connection tests
@pytest.mark.asyncio
async def test_database_connection():
    """Test basic database connectivity."""
    result = await test_connection()
    assert result is True


@pytest.mark.asyncio
async def test_table_existence():
    """Test that all required tables exist."""
    async with engine.connect() as conn:
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'loans', 'ecl_segment_calculation', 'permissions')
            ORDER BY table_name
        """)
        result = await conn.execute(query)
        tables = [row[0] for row in result.fetchall()]
        
        assert len(tables) == 4
        assert 'users' in tables
        assert 'loans' in tables
        assert 'ecl_segment_calculation' in tables
        assert 'permissions' in tables


# User CRUD tests
@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession, test_user_data):
    """Test user creation."""
    # Hash password
    hashed_password = hash_password(test_user_data["password"])
    
    # Create user
    user = await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hashed_password,
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    assert user.username == test_user_data["username"]
    assert user.email == test_user_data["email"]
    assert user.role == test_user_data["role"]
    assert user.user_id is not None


@pytest.mark.asyncio
async def test_get_user_by_username(db_session: AsyncSession, test_user_data):
    """Test retrieving user by username."""
    # First create a user
    hashed_password = hash_password(test_user_data["password"])
    created_user = await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hashed_password,
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    # Then retrieve it
    retrieved_user = await crud.get_user_by_username(db_session, test_user_data["username"])
    
    assert retrieved_user is not None
    assert retrieved_user.user_id == created_user.user_id
    assert retrieved_user.username == test_user_data["username"]


@pytest.mark.asyncio
async def test_authenticate_user(db_session: AsyncSession, test_user_data):
    """Test user authentication."""
    # Create user with hashed password
    hashed_password = hash_password(test_user_data["password"])
    await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hashed_password,
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    # Test authentication with correct password
    from auth.auth import authenticate_user
    authenticated = await authenticate_user(
        db_session,
        test_user_data["username"],
        test_user_data["password"]
    )
    
    assert authenticated is not None
    assert authenticated.username == test_user_data["username"]
    
    # Test authentication with wrong password
    wrong_auth = await authenticate_user(
        db_session,
        test_user_data["username"],
        "WrongPassword"
    )
    
    assert wrong_auth is None


# Password hashing tests
def test_password_hashing():
    """Test password hashing and verification."""
    password = "TestPassword123"
    
    # Hash password
    hashed = hash_password(password)
    
    # Verify correct password
    assert verify_password(password, hashed) is True
    
    # Verify wrong password
    assert verify_password("WrongPassword", hashed) is False
    
    # Ensure hashes are different
    hashed2 = hash_password(password)
    assert hashed != hashed2


# JWT token tests
def test_jwt_creation_and_decode():
    """Test JWT token creation and decoding."""
    # Create token data
    data = {"sub": "123", "role": "Analyst"}
    
    # Create token
    token = create_access_token(data)
    
    assert token is not None
    assert isinstance(token, str)
    
    # Decode token
    decoded = decode_access_token(token)
    
    assert decoded["sub"] == "123"
    assert decoded["role"] == "Analyst"
    assert "exp" in decoded


def test_jwt_expiration():
    """Test JWT token expiration."""
    # Create token with short expiration
    data = {"sub": "123"}
    expires_delta = timedelta(seconds=-1)  # Already expired
    
    token = create_access_token(data, expires_delta)
    
    # Should raise exception when decoding expired token
    try:
        decode_access_token(token)
        assert False, "Should have raised exception for expired token"
    except:
        assert True


# Loan CRUD tests
@pytest.mark.asyncio
async def test_bulk_insert_loans(db_session: AsyncSession, test_user_data, test_loan_data):
    """Test bulk loan insertion."""
    # First create a user
    user = await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hash_password(test_user_data["password"]),
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    # Prepare loan data
    loans_data = [
        {**test_loan_data, "user_id": user.user_id},
        {**test_loan_data, "user_id": user.user_id, "loan_amount": 15000.0},
        {**test_loan_data, "user_id": user.user_id, "loan_amount": 20000.0}
    ]
    
    # Bulk insert
    loan_ids = await crud.bulk_insert_loans(db_session, loans_data)
    
    assert len(loan_ids) == 3
    assert all(isinstance(loan_id, int) for loan_id in loan_ids)


@pytest.mark.asyncio
async def test_get_loans_by_user(db_session: AsyncSession, test_user_data, test_loan_data):
    """Test retrieving loans by user ID."""
    # Create user and loans
    user = await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hash_password(test_user_data["password"]),
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    loans_data = [
        {**test_loan_data, "user_id": user.user_id},
        {**test_loan_data, "user_id": user.user_id, "loan_intent": "EDUCATION"}
    ]
    
    await crud.bulk_insert_loans(db_session, loans_data)
    
    # Retrieve loans
    user_loans = await crud.get_loans_by_user_id(db_session, user.user_id)
    
    assert len(user_loans) == 2
    assert all(loan.user_id == user.user_id for loan in user_loans)


# ECL CRUD tests
@pytest.mark.asyncio
async def test_ecl_calculations_crud(db_session: AsyncSession, test_user_data):
    """Test ECL calculation CRUD operations."""
    # Create user
    user = await crud.create_user(
        db_session,
        username=test_user_data["username"],
        password_hash=hash_password(test_user_data["password"]),
        email=test_user_data["email"],
        role=test_user_data["role"]
    )
    
    # Create ECL calculations
    ecl_data = [
        {
            "user_id": user.user_id,
            "loan_id": None,  # Aggregate calculation
            "segment_name": "loan_intent",
            "segment_value": "PERSONAL",
            "pd_value": 0.15,
            "lgd_value": 0.35,
            "ead_value": 10000.0,
            "ecl_value": 525.0
        },
        {
            "user_id": user.user_id,
            "loan_id": None,
            "segment_name": "age_group",
            "segment_value": "young",
            "pd_value": 0.12,
            "lgd_value": 0.35,
            "ead_value": 8000.0,
            "ecl_value": 336.0
        }
    ]
    
    # Insert ECL calculations
    ecl_ids = await crud.bulk_insert_ecl_calculations(db_session, ecl_data)
    
    assert len(ecl_ids) == 2
    
    # Retrieve ECL by user
    user_ecl = await crud.get_ecl_by_user(db_session, user.user_id)
    
    assert len(user_ecl) >= 2


# Permission CRUD tests
@pytest.mark.asyncio
async def test_permissions_crud(db_session: AsyncSession, test_user_data):
    """Test permission CRUD operations."""
    # Create an analyst user
    analyst = await crud.create_user(
        db_session,
        username="analyst1",
        password_hash=hash_password("password123"),
        email="analyst1@example.com",
        role="Analyst"
    )
    
    # Create a CRO user
    cro = await crud.create_user(
        db_session,
        username="cro1",
        password_hash=hash_password("password123"),
        email="cro@example.com",
        role="CRO"
    )
    
    # Grant permissions to analyst
    permission = await crud.create_permission(
        db_session,
        user_id=analyst.user_id,
        segment_name="loan_intent",
        permission_type="read"
    )
    
    assert permission.user_id == analyst.user_id
    assert permission.segment_name == "loan_intent"
    assert permission.permission_type == "read"
    
    # Check permissions
    has_perm = await crud.check_user_segment_access(
        db_session,
        analyst.user_id,
        "loan_intent",
        "read"
    )
    assert has_perm is True
    
    has_perm_write = await crud.check_user_segment_access(
        db_session,
        analyst.user_id,
        "loan_intent",
        "write"
    )
    assert has_perm_write is False
    
    # CRO should have access without explicit permission
    cro_has_perm = await crud.check_user_segment_access(
        db_session,
        cro.user_id,
        "loan_intent",
        "read"
    )
    assert cro_has_perm is True


# Run tests
if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_database_connection())
    print("✅ Database connection test passed")
    
    # Run other tests with pytest
    pytest.main([__file__, "-v"])
