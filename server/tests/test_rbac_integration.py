"""
RBAC (Role-Based Access Control) Integration Tests.

Tests permission scenarios, segment access control, and filtering.
"""

import asyncio
import pytest
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from db.database import engine
from db import crud
from db.models import User, ECLSegmentCalculation
from auth.auth import hash_password
from auth.rbac import (
    check_segment_permission,
    filter_ecl_by_permissions,
    filter_segments_dict_by_permissions,
    get_user_accessible_segments,
    check_bulk_permissions
)


# Test fixtures
@pytest.fixture
async def db_session():
    """Get a test database session."""
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_users(db_session: AsyncSession):
    """Create test users with different roles."""
    # Create CRO user
    cro = await crud.create_user(
        db_session,
        username="test_cro",
        password_hash=hash_password("password123"),
        email="cro@test.com",
        role="CRO"
    )
    
    # Create Analyst users
    analyst1 = await crud.create_user(
        db_session,
        username="test_analyst1",
        password_hash=hash_password("password123"),
        email="analyst1@test.com",
        role="Analyst"
    )
    
    analyst2 = await crud.create_user(
        db_session,
        username="test_analyst2",
        password_hash=hash_password("password123"),
        email="analyst2@test.com",
        role="Analyst"
    )
    
    return {
        "cro": cro,
        "analyst1": analyst1,
        "analyst2": analyst2
    }


@pytest.fixture
async def test_permissions(db_session: AsyncSession, test_users):
    """Create test permissions for analysts."""
    users = test_users
    
    # Give analyst1 read access to loan_intent and age_group
    await crud.create_permission(
        db_session,
        user_id=users["analyst1"].user_id,
        segment_name="loan_intent",
        permission_type="read"
    )
    
    await crud.create_permission(
        db_session,
        user_id=users["analyst1"].user_id,
        segment_name="age_group",
        permission_type="read"
    )
    
    # Give analyst2 read access to person_gender only
    await crud.create_permission(
        db_session,
        user_id=users["analyst2"].user_id,
        segment_name="person_gender",
        permission_type="read"
    )
    
    # Give analyst2 write access to person_education
    await crud.create_permission(
        db_session,
        user_id=users["analyst2"].user_id,
        segment_name="person_education",
        permission_type="write"
    )
    
    return users


@pytest.fixture
async def test_ecl_calculations(db_session: AsyncSession, test_users):
    """Create test ECL calculations."""
    users = test_users
    
    ecl_data = [
        # CRO's calculations
        {
            "user_id": users["cro"].user_id,
            "segment_name": "loan_intent",
            "segment_value": "PERSONAL",
            "pd_value": 0.15,
            "lgd_value": 0.35,
            "ead_value": 10000.0,
            "ecl_value": 525.0
        },
        {
            "user_id": users["cro"].user_id,
            "segment_name": "age_group",
            "segment_value": "senior_citizen",
            "pd_value": 0.25,
            "lgd_value": 0.40,
            "ead_value": 12000.0,
            "ecl_value": 1200.0
        },
        {
            "user_id": users["cro"].user_id,
            "segment_name": "person_gender",
            "segment_value": "female",
            "pd_value": 0.10,
            "lgd_value": 0.35,
            "ead_value": 8000.0,
            "ecl_value": 280.0
        },
        {
            "user_id": users["cro"].user_id,
            "segment_name": "person_education",
            "segment_value": "Master",
            "pd_value": 0.08,
            "lgd_value": 0.30,
            "ead_value": 15000.0,
            "ecl_value": 360.0
        }
    ]
    
    await crud.bulk_insert_ecl_calculations(db_session, ecl_data)
    
    # Get the created ECL calculations
    ecl_records = await crud.get_ecl_by_user(db_session, users["cro"].user_id)
    
    return ecl_records


# Permission checking tests
@pytest.mark.asyncio
async def test_cro_has_all_permissions(db_session: AsyncSession, test_permissions):
    """Test that CRO users have access to all segments."""
    users = test_permissions
    cro = users["cro"]
    
    # Check various segments
    segments = ["loan_intent", "age_group", "person_gender", "person_education", "person_home_ownership"]
    
    for segment in segments:
        has_read = await check_segment_permission(cro, segment, "read", db_session)
        has_write = await check_segment_permission(cro, segment, "write", db_session)
        
        assert has_read is True, f"CRO should have read access to {segment}"
        assert has_write is True, f"CRO should have write access to {segment}"


@pytest.mark.asyncio
async def test_analyst_limited_permissions(db_session: AsyncSession, test_permissions):
    """Test that Analysts have only permitted access."""
    users = test_permissions
    analyst1 = users["analyst1"]
    analyst2 = users["analyst2"]
    
    # Test analyst1 permissions
    assert await check_segment_permission(analyst1, "loan_intent", "read", db_session) is True
    assert await check_segment_permission(analyst1, "loan_intent", "write", db_session) is False
    assert await check_segment_permission(analyst1, "age_group", "read", db_session) is True
    assert await check_segment_permission(analyst1, "person_gender", "read", db_session) is False
    
    # Test analyst2 permissions
    assert await check_segment_permission(analyst2, "person_gender", "read", db_session) is True
    assert await check_segment_permission(analyst2, "person_education", "write", db_session) is True
    assert await check_segment_permission(analyst2, "loan_intent", "read", db_session) is False


# ECL filtering tests
@pytest.mark.asyncio
async def test_filter_ecl_by_permissions(
    db_session: AsyncSession,
    test_permissions,
    test_ecl_calculations
):
    """Test filtering ECL results based on user permissions."""
    users = test_permissions
    ecl_records = test_ecl_calculations
    
    # CRO should see all records
    cro_filtered = await filter_ecl_by_permissions(users["cro"], ecl_records, db_session)
    assert len(cro_filtered) == len(ecl_records)
    
    # Analyst1 should see only loan_intent and age_group
    analyst1_filtered = await filter_ecl_by_permissions(users["analyst1"], ecl_records, db_session)
    analyst1_segments = [r.segment_name for r in analyst1_filtered]
    assert len(analyst1_filtered) == 2
    assert "loan_intent" in analyst1_segments
    assert "age_group" in analyst1_segments
    
    # Analyst2 should see only person_gender and person_education
    analyst2_filtered = await filter_ecl_by_permissions(users["analyst2"], ecl_records, db_session)
    analyst2_segments = [r.segment_name for r in analyst2_filtered]
    assert len(analyst2_filtered) == 2
    assert "person_gender" in analyst2_segments
    assert "person_education" in analyst2_segments


@pytest.mark.asyncio
async def test_filter_segments_dict_by_permissions(
    db_session: AsyncSession,
    test_permissions
):
    """Test filtering dictionary of segments based on permissions."""
    users = test_permissions
    
    # Create a test segments dictionary
    segments_dict = {
        "loan_intent": [{"segment": "PERSONAL", "ecl": 500}],
        "age_group": [{"segment": "young", "ecl": 300}],
        "person_gender": [{"segment": "male", "ecl": 400}],
        "person_education": [{"segment": "Bachelor", "ecl": 350}],
        "person_home_ownership": [{"segment": "RENT", "ecl": 450}]
    }
    
    # CRO should get all segments
    cro_filtered = await filter_segments_dict_by_permissions(
        users["cro"], segments_dict, db_session
    )
    assert len(cro_filtered) == len(segments_dict)
    
    # Analyst1 should get only loan_intent and age_group
    analyst1_filtered = await filter_segments_dict_by_permissions(
        users["analyst1"], segments_dict, db_session
    )
    assert len(analyst1_filtered) == 2
    assert "loan_intent" in analyst1_filtered
    assert "age_group" in analyst1_filtered
    assert "person_gender" not in analyst1_filtered


# Accessible segments tests
@pytest.mark.asyncio
async def test_get_user_accessible_segments(
    db_session: AsyncSession,
    test_permissions
):
    """Test retrieving list of accessible segments for users."""
    users = test_permissions
    
    # CRO should have access to all segment types
    cro_segments = await get_user_accessible_segments(users["cro"], db_session)
    # Should include all SEGMENT_COLUMNS from config
    assert "loan_intent" in cro_segments
    assert "age_group" in cro_segments
    assert "person_gender" in cro_segments
    assert "person_education" in cro_segments
    assert "person_home_ownership" in cro_segments
    
    # Analyst1 should have limited access
    analyst1_segments = await get_user_accessible_segments(users["analyst1"], db_session)
    assert len(analyst1_segments) == 2
    assert "loan_intent" in analyst1_segments
    assert "age_group" in analyst1_segments
    
    # Analyst2 should have different limited access
    analyst2_segments = await get_user_accessible_segments(users["analyst2"], db_session)
    assert len(analyst2_segments) == 2
    assert "person_gender" in analyst2_segments
    assert "person_education" in analyst2_segments


# Bulk permission checking tests
@pytest.mark.asyncio
async def test_check_bulk_permissions(
    db_session: AsyncSession,
    test_permissions
):
    """Test checking permissions for multiple segments at once."""
    users = test_permissions
    
    segment_names = ["loan_intent", "age_group", "person_gender", "person_education"]
    
    # Check for analyst1
    analyst1_perms = await check_bulk_permissions(
        users["analyst1"], segment_names, "read", db_session
    )
    
    assert analyst1_perms["loan_intent"] is True
    assert analyst1_perms["age_group"] is True
    assert analyst1_perms["person_gender"] is False
    assert analyst1_perms["person_education"] is False
    
    # Check for CRO (all should be True)
    cro_perms = await check_bulk_permissions(
        users["cro"], segment_names, "read", db_session
    )
    
    assert all(cro_perms.values()), "CRO should have access to all segments"


# Edge case tests
@pytest.mark.asyncio
async def test_no_permissions_analyst(db_session: AsyncSession):
    """Test analyst with no permissions."""
    # Create analyst with no permissions
    no_perm_analyst = await crud.create_user(
        db_session,
        username="no_perm_analyst",
        password_hash=hash_password("password123"),
        email="noperm@test.com",
        role="Analyst"
    )
    
    # Should have no access
    accessible = await get_user_accessible_segments(no_perm_analyst, db_session)
    assert len(accessible) == 0
    
    # Should not have access to any segment
    has_access = await check_segment_permission(
        no_perm_analyst, "loan_intent", "read", db_session
    )
    assert has_access is False


@pytest.mark.asyncio
async def test_permission_inheritance(
    db_session: AsyncSession,
    test_permissions
):
    """Test that write permission doesn't imply read permission."""
    users = test_permissions
    analyst2 = users["analyst2"]
    
    # Analyst2 has write permission on person_education
    has_write = await check_segment_permission(
        analyst2, "person_education", "write", db_session
    )
    assert has_write is True
    
    # But no explicit read permission
    # However, since they have write, they should be able to see it in the list
    accessible = await get_user_accessible_segments(analyst2, db_session)
    assert "person_education" in accessible


# Run tests
if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
