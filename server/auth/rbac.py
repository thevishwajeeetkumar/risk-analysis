"""
Role-Based Access Control (RBAC) utilities.

Provides permission checking and filtering functions for segment-level access control.
"""

import logging
from typing import List, Dict, Any, Callable

from fastapi import Depends, HTTPException, status
from sqlalchemy import select, func, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from db.database import get_db
from db import crud
from db.models import User, ECLSegmentCalculation, Permission
from auth.auth import get_current_user
from core.config import (
    SEGMENT_COLUMNS,
    RBAC_AUTO_GRANT_DEFAULTS,
    RBAC_FUZZY_MATCHING,
    RBAC_AUTO_GRANT_SEGMENTS,
)

# Initialize logger
logger = logging.getLogger(__name__)


async def check_segment_permission(
    user: User,
    segment_name: str,
    permission_type: str,
    db: AsyncSession
) -> bool:
    """
    Check if a user has permission to access a segment.
    
    Args:
        user: User object to check permissions for
        segment_name: Name of the segment (e.g., "loan_intent", "age_group")
        permission_type: Type of permission ("read" or "write")
        db: Database session
        
    Returns:
        True if user has permission, False otherwise
    """
    # CROs have access to all segments
    if user.role == "CRO":
        return True
    
    # For Analysts, check specific permissions
    return await crud.has_permission(db, user.user_id, segment_name, permission_type)


def require_segment_read(segment_name: str) -> Callable:
    """
    FastAPI dependency factory that requires read access to a segment.
    
    Args:
        segment_name: Name of the segment to check access for
        
    Returns:
        Dependency function that validates segment access
    """
    async def segment_permission_dependency(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ):
        has_permission = await check_segment_permission(
            current_user,
            segment_name,
            "read",
            db
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No read access to segment: {segment_name}"
            )
        
        return current_user
    
    return segment_permission_dependency


def require_segment_write(segment_name: str) -> Callable:
    """
    FastAPI dependency factory that requires write access to a segment.
    
    Args:
        segment_name: Name of the segment to check access for
        
    Returns:
        Dependency function that validates segment access
    """
    async def segment_permission_dependency(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ):
        has_permission = await check_segment_permission(
            current_user,
            segment_name,
            "write",
            db
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No write access to segment: {segment_name}"
            )
        
        return current_user
    
    return segment_permission_dependency


async def filter_ecl_by_permissions(
    user: User,
    ecl_results: List[ECLSegmentCalculation],
    db: AsyncSession
) -> List[ECLSegmentCalculation]:
    """
    Filter ECL results based on user's permissions.
    
    Args:
        user: User to filter results for
        ecl_results: List of ECL calculation results
        db: Database session
        
    Returns:
        Filtered list containing only segments user has access to
    """
    # CROs see all results
    if user.role == "CRO":
        return ecl_results
    
    # For Analysts, filter based on permissions
    filtered_results = []
    
    for ecl in ecl_results:
        has_permission = await check_segment_permission(
            user,
            ecl.segment_name,
            "read",
            db
        )
        if has_permission:
            filtered_results.append(ecl)
    
    return filtered_results


async def filter_segments_dict_by_permissions(
    user: User,
    segments_dict: Dict[str, Any],
    db: AsyncSession
) -> Dict[str, Any]:
    """
    Filter a dictionary of segments based on user's permissions.
    
    Enhanced with:
    - Fuzzy matching for case/spelling variations
    - Auto-grant of default permissions for core segments (Analysts only)
    - Comprehensive logging for audit trail
    
    Args:
        user: User to filter results for
        segments_dict: Dictionary mapping segment names to data
        db: Database session
        
    Returns:
        Filtered dictionary containing only segments user has access to
    """
    # CROs see all segments
    if user.role == "CRO":
        return segments_dict
    
    # Define core segments that Analysts should have default read access to
    CORE_SEGMENTS = set(RBAC_AUTO_GRANT_SEGMENTS)
    
    # For Analysts, filter based on permissions with fuzzy matching and defaults
    filtered_dict = {}
    
    for segment_type, segment_data in segments_dict.items():
        has_perm = False
        
        # Check DB with optional fuzzy SQL LIKE matching
        try:
            if RBAC_FUZZY_MATCHING:
                # Normalize segment type for fuzzy matching
                normalized_type = segment_type.lower().replace('_', '-').strip()
                result = await db.execute(
                    select(Permission).where(
                        Permission.user_id == user.user_id,
                        Permission.permission_type == "read",
                        func.lower(Permission.segment_name).like(f"%{normalized_type}%")
                    )
                )
            else:
                # Exact match only
                result = await db.execute(
                    select(Permission).where(
                        Permission.user_id == user.user_id,
                        Permission.permission_type == "read",
                        Permission.segment_name == segment_type
                    )
                )
            has_perm = bool(result.scalar_one_or_none())
        except Exception as e:
            logger.error(f"[RBAC] Error checking permission for {segment_type}: {e}")
            # Continue without permission if DB error
        
        # Fallback: Auto-grant defaults for Analysts on core segments (if enabled)
        if (
            not has_perm
            and RBAC_AUTO_GRANT_DEFAULTS
            and user.role == "Analyst"
            and segment_type in CORE_SEGMENTS
        ):
            logger.info(f"[RBAC] Auto-granting default 'read' for {user.username} on core segment '{segment_type}'")
            try:
                # Insert new permission
                await db.execute(
                    insert(Permission).values(
                        user_id=user.user_id,
                        segment_name=segment_type,
                        permission_type="read"
                    )
                )
                await db.commit()
                has_perm = True
                logger.info(f"[RBAC] Successfully auto-granted permission for {segment_type}")
            except IntegrityError as e:
                await db.rollback()
                message = str(e.orig) if getattr(e, "orig", None) else str(e)
                if "permissions_segment_name_check" in message:
                    logger.warning(
                        "[RBAC] Auto-grant blocked by constraint for %s on segment '%s'. "
                        "Verify database CHECK constraint allows this segment.",
                        user.username,
                        segment_type,
                    )
                    has_perm = False
                else:
                    logger.warning(
                        "[RBAC] Integrity error auto-granting %s for %s: %s. Assuming permission exists.",
                        segment_type,
                        user.username,
                        message,
                    )
                    has_perm = True
            except Exception as e:
                logger.warning(
                    "[RBAC] Failed to auto-grant %s for %s: %s",
                    segment_type,
                    user.username,
                    e,
                )
                await db.rollback()
                has_perm = False
        
        if has_perm:
            filtered_dict[segment_type] = segment_data
    
    return filtered_dict


async def get_user_accessible_segments(
    user: User,
    db: AsyncSession
) -> List[str]:
    """
    Get list of segments a user has access to.
    
    Args:
        user: User to get accessible segments for
        db: Database session
        
    Returns:
        List of segment names user can access
    """
    # CROs have access to all segments
    if user.role == "CRO":
        # Return all possible segment types from config
        from core.config import SEGMENT_COLUMNS
        return SEGMENT_COLUMNS
    
    # For Analysts, get their permissions
    permissions = await crud.get_user_permissions(db, user.user_id)
    
    # Extract unique segment names
    accessible_segments = list(set(p.segment_name for p in permissions))
    
    return accessible_segments


async def check_bulk_permissions(
    user: User,
    segment_names: List[str],
    permission_type: str,
    db: AsyncSession
) -> Dict[str, bool]:
    """
    Check permissions for multiple segments at once.
    
    Args:
        user: User to check permissions for
        segment_names: List of segment names to check
        permission_type: Type of permission ("read" or "write")
        db: Database session
        
    Returns:
        Dictionary mapping segment names to permission status
    """
    # CROs have access to all segments
    if user.role == "CRO":
        return {segment: True for segment in segment_names}
    
    # For Analysts, check each segment
    permissions_map = {}
    
    for segment_name in segment_names:
        has_permission = await crud.has_permission(
            db,
            user.user_id,
            segment_name,
            permission_type
        )
        permissions_map[segment_name] = has_permission
    
    return permissions_map


def require_any_segment_read(segment_names: List[str]) -> Callable:
    """
    FastAPI dependency that requires read access to at least one of the segments.
    
    Args:
        segment_names: List of segment names to check
        
    Returns:
        Dependency function that validates access to at least one segment
    """
    async def any_segment_permission_dependency(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ):
        # CROs have access to all
        if current_user.role == "CRO":
            return current_user
        
        # Check if user has access to any of the segments
        for segment_name in segment_names:
            has_permission = await crud.has_permission(
                db,
                current_user.user_id,
                segment_name,
                "read"
            )
            if has_permission:
                return current_user
        
        # No access to any segment
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to any of the segments: {', '.join(segment_names)}"
        )
    
    return any_segment_permission_dependency

