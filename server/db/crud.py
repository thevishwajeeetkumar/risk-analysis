"""
CRUD operations for database models using async SQLAlchemy.

Provides Create, Read, Update, Delete operations for:
- Users (authentication and user management)
- Loans (loan data storage)
- ECL Segment Calculations (ECL results)
- Permissions (RBAC)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import select, and_, or_, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.models import User, Loan, ECLSegmentCalculation, Permission


# ==================== User CRUD ====================

async def create_user(
    db: AsyncSession,
    username: str,
    password_hash: str,
    email: str,
    role: str = "Analyst"
) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        username: Unique username
        password_hash: Hashed password
        email: User email
        role: User role (Analyst or CRO)
        
    Returns:
        Created User object
        
    Raises:
        Exception: If user creation fails (e.g., duplicate username/email)
    """
    try:
        user = User(
            username=username,
            password_hash=password_hash,
            email=email,
            role=role
        )
        db.add(user)
        # Flush to get the user_id (but don't commit - let get_db dependency handle it)
        await db.flush()
        await db.refresh(user)
        print(f"[SUCCESS] User created in database: {username} (ID: {user.user_id})")
        # Don't commit here - let get_db dependency handle the commit
        return user
    except Exception as e:
        # Don't rollback here - let get_db dependency handle it
        # But log the error for debugging
        print(f"[ERROR] Database error creating user: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # Re-raise to let get_db handle rollback
        raise


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get user by username."""
    try:
        result = await db.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        if user:
            print(f"[SUCCESS] Found user in database: {user.username} (ID: {user.user_id})")
        return user
    except Exception as e:
        print(f"[ERROR] Error fetching user by username: {str(e)}")
        raise


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID."""
    try:
        # Ensure user_id is an integer
        if not isinstance(user_id, int):
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                print(f"[ERROR] Invalid user_id type: {type(user_id)}, value: {user_id}")
                return None
        
        result = await db.execute(
            select(User).where(User.user_id == user_id)
        )
        user = result.scalar_one_or_none()
        if user:
            print(f"[SUCCESS] Found user by ID: {user.username} (ID: {user.user_id})")
        return user
    except Exception as e:
        print(f"[ERROR] Error fetching user by ID: {str(e)}")
        raise


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email."""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def authenticate_user(
    db: AsyncSession,
    username: str,
    password_hash: str
) -> Optional[User]:
    """Authenticate user by username and password hash."""
    result = await db.execute(
        select(User).where(
            and_(
                User.username == username,
                User.password_hash == password_hash
            )
        )
    )
    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> List[User]:
    """List all users with pagination."""
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    return result.scalars().all()


# ==================== Loan CRUD ====================

async def create_loan(db: AsyncSession, **loan_data) -> Loan:
    """Create a single loan record."""
    from db.schema_check import ensure_loans_schema

    await ensure_loans_schema(db)

    loan = Loan(**loan_data)
    db.add(loan)
    await db.flush()  # Use flush instead of commit (get_db handles commit)
    await db.refresh(loan)
    return loan


async def bulk_insert_loans(
    db: AsyncSession,
    loans_data: List[Dict[str, Any]]
) -> List[int]:
    """Bulk insert multiple loans efficiently."""
    from db.schema_check import ensure_loans_schema

    await ensure_loans_schema(db)

    try:
        loans = [Loan(**loan_data) for loan_data in loans_data]

        db.add_all(loans)
        await db.flush()  # Flush to get IDs without committing

        loan_ids = [loan.loan_id for loan in loans]

        return loan_ids
    except Exception as e:
        print(f"[ERROR] Error bulk inserting loans: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


async def get_loan_by_id(db: AsyncSession, loan_id: int) -> Optional[Loan]:
    """Get loan by ID."""
    result = await db.execute(
        select(Loan)
        .options(selectinload(Loan.user))
        .where(Loan.loan_id == loan_id)
    )
    return result.scalar_one_or_none()


async def get_loans_by_user_id(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[Loan]:
    """Get loans by user ID with pagination."""
    result = await db.execute(
        select(Loan)
        .where(Loan.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .order_by(Loan.created_at.desc())
    )
    return result.scalars().all()


async def get_loans_with_filters(
    db: AsyncSession,
    user_id: Optional[int] = None,
    loan_intent: Optional[str] = None,
    loan_status: Optional[int] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Loan]:
    """Get loans with various filters."""
    query = select(Loan)
    
    # Build filters
    filters = []
    if user_id is not None:
        filters.append(Loan.user_id == user_id)
    if loan_intent is not None:
        filters.append(Loan.loan_intent == loan_intent)
    if loan_status is not None:
        filters.append(Loan.loan_status == loan_status)
    if min_amount is not None:
        filters.append(Loan.loan_amount >= min_amount)
    if max_amount is not None:
        filters.append(Loan.loan_amount <= max_amount)
    
    if filters:
        query = query.where(and_(*filters))
    
    query = query.offset(skip).limit(limit).order_by(Loan.created_at.desc())
    
    result = await db.execute(query)
    return result.scalars().all()


async def count_loans_by_user(db: AsyncSession, user_id: int) -> int:
    """Count total loans for a user."""
    result = await db.execute(
        select(func.count(Loan.loan_id)).where(Loan.user_id == user_id)
    )
    return result.scalar() or 0


# ==================== ECL Segment Calculation CRUD ====================

async def create_ecl_calculation(
    db: AsyncSession,
    **ecl_data
) -> ECLSegmentCalculation:
    """Create a single ECL calculation record."""
    ecl = ECLSegmentCalculation(**ecl_data)
    db.add(ecl)
    await db.commit()
    await db.refresh(ecl)
    return ecl


async def bulk_insert_ecl_calculations(
    db: AsyncSession,
    ecl_records: List[Dict[str, Any]]
) -> List[int]:
    """
    Bulk insert multiple ECL calculation records.
    
    Returns list of created ECL IDs.
    """
    # Create ECL objects
    ecls = [ECLSegmentCalculation(**ecl_data) for ecl_data in ecl_records]
    
    # Bulk insert
    db.add_all(ecls)
    await db.commit()
    
    # Get IDs of inserted records
    ecl_ids = [ecl.ecl_id for ecl in ecls]
    return ecl_ids


async def get_ecl_by_user(
    db: AsyncSession,
    user_id: int,
    file_id: Optional[str] = None,
    segment_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[ECLSegmentCalculation]:
    """Get ECL calculations by user with optional filters."""
    query = select(ECLSegmentCalculation).where(
        ECLSegmentCalculation.user_id == user_id
    )
    
    if segment_type:
        query = query.where(ECLSegmentCalculation.segment_name == segment_type)
    
    # Note: file_id filtering would require adding file_id to the model
    # For now, we'll use created_at for grouping calculations
    
    query = query.offset(skip).limit(limit).order_by(
        ECLSegmentCalculation.created_at.desc(),
        ECLSegmentCalculation.ecl_value.desc()
    )
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_ecl_by_segment(
    db: AsyncSession,
    segment_name: str,
    segment_value: Optional[str] = None,
    user_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100
) -> List[ECLSegmentCalculation]:
    """Get ECL calculations by segment."""
    query = select(ECLSegmentCalculation).where(
        ECLSegmentCalculation.segment_name == segment_name
    )
    
    if segment_value:
        query = query.where(ECLSegmentCalculation.segment_value == segment_value)
    
    if user_id:
        query = query.where(ECLSegmentCalculation.user_id == user_id)
    
    query = query.offset(skip).limit(limit).order_by(
        ECLSegmentCalculation.ecl_value.desc()
    )
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_ecl_history(
    db: AsyncSession,
    user_id: int,
    days: int = 30,
    segment_name: Optional[str] = None
) -> List[ECLSegmentCalculation]:
    """Get ECL calculation history for the last N days."""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    query = select(ECLSegmentCalculation).where(
        and_(
            ECLSegmentCalculation.user_id == user_id,
            ECLSegmentCalculation.created_at >= cutoff_date
        )
    )
    
    if segment_name:
        query = query.where(ECLSegmentCalculation.segment_name == segment_name)
    
    query = query.order_by(ECLSegmentCalculation.created_at.desc())
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_latest_ecl_by_segment_type(
    db: AsyncSession,
    user_id: int,
    segment_names: List[str]
) -> Dict[str, List[ECLSegmentCalculation]]:
    """
    Get the latest ECL calculations for each segment type.
    
    Returns a dictionary mapping segment_name to list of ECL records.
    """
    results = {}
    
    for segment_name in segment_names:
        # Get the latest batch of calculations for this segment type
        # We assume calculations done at the same time belong together
        subquery = (
            select(func.max(ECLSegmentCalculation.created_at))
            .where(
                and_(
                    ECLSegmentCalculation.user_id == user_id,
                    ECLSegmentCalculation.segment_name == segment_name
                )
            )
        )
        
        result = await db.execute(
            select(ECLSegmentCalculation).where(
                and_(
                    ECLSegmentCalculation.user_id == user_id,
                    ECLSegmentCalculation.segment_name == segment_name,
                    ECLSegmentCalculation.created_at == subquery
                )
            ).order_by(ECLSegmentCalculation.ecl_value.desc())
        )
        
        results[segment_name] = result.scalars().all()
    
    return results


# ==================== Permission CRUD ====================

async def create_permission(
    db: AsyncSession,
    user_id: int,
    segment_name: str,
    permission_type: str = "read"
) -> Permission:
    """Create a new permission."""
    permission = Permission(
        user_id=user_id,
        segment_name=segment_name,
        permission_type=permission_type
    )
    db.add(permission)
    await db.commit()
    await db.refresh(permission)
    return permission


async def get_user_permissions(
    db: AsyncSession,
    user_id: int
) -> List[Permission]:
    """Get all permissions for a user."""
    result = await db.execute(
        select(Permission)
        .where(Permission.user_id == user_id)
        .order_by(Permission.segment_name)
    )
    return result.scalars().all()


async def check_user_segment_access(
    db: AsyncSession,
    user_id: int,
    segment_name: str,
    permission_type: str = "read"
) -> bool:
    """Check if user has specific permission for a segment."""
    # First check if user is CRO (has access to everything)
    user = await get_user_by_id(db, user_id)
    if user and user.role == "CRO":
        return True
    
    # Check specific permission
    result = await db.execute(
        select(Permission).where(
            and_(
                Permission.user_id == user_id,
                Permission.segment_name == segment_name,
                Permission.permission_type == permission_type
            )
        )
    )
    
    return result.scalar_one_or_none() is not None


async def has_permission(
    db: AsyncSession,
    user_id: int,
    segment_name: str,
    permission_type: str = "read"
) -> bool:
    """Alias for check_user_segment_access."""
    return await check_user_segment_access(db, user_id, segment_name, permission_type)


async def grant_permissions(
    db: AsyncSession,
    user_id: int,
    permissions: List[Dict[str, str]]
) -> List[Permission]:
    """
    Grant multiple permissions to a user.
    
    Args:
        user_id: User to grant permissions to
        permissions: List of dicts with 'segment_name' and 'permission_type'
    """
    created_permissions = []
    
    for perm_data in permissions:
        # Check if permission already exists
        exists = await check_user_segment_access(
            db,
            user_id,
            perm_data['segment_name'],
            perm_data['permission_type']
        )
        
        if not exists:
            permission = await create_permission(
                db,
                user_id,
                perm_data['segment_name'],
                perm_data['permission_type']
            )
            created_permissions.append(permission)
    
    return created_permissions


async def revoke_permission(
    db: AsyncSession,
    user_id: int,
    segment_name: str,
    permission_type: str = "read"
) -> bool:
    """Revoke a specific permission from a user."""
    result = await db.execute(
        delete(Permission).where(
            and_(
                Permission.user_id == user_id,
                Permission.segment_name == segment_name,
                Permission.permission_type == permission_type
            )
        )
    )
    await db.commit()
    return result.rowcount > 0


async def get_users_by_segment_access(
    db: AsyncSession,
    segment_name: str,
    permission_type: str = "read"
) -> List[User]:
    """Get all users who have access to a specific segment."""
    # Get all users with explicit permissions
    result = await db.execute(
        select(User)
        .join(Permission, User.user_id == Permission.user_id)
        .where(
            and_(
                Permission.segment_name == segment_name,
                Permission.permission_type == permission_type
            )
        )
    )
    users_with_permission = result.scalars().all()
    
    # Also get all CROs (they have implicit access)
    cro_result = await db.execute(
        select(User).where(User.role == "CRO")
    )
    cros = cro_result.scalars().all()
    
    # Combine and deduplicate
    all_users = list({user.user_id: user for user in users_with_permission + cros}.values())
    
    return all_users


# ==================== RAG Document CRUD ====================

async def store_rag_document(
    db: AsyncSession,
    doc_id: str,
    content: str,
    metadata: str,
    user_id: Optional[int] = None
) -> "RAGDocument":
    """
    Store a RAG document in the database.
    
    Args:
        db: Database session
        doc_id: Unique document identifier
        content: Document content
        metadata: JSON-encoded metadata
        user_id: Optional user ID
        
    Returns:
        Created or updated RAGDocument
    """
    from db.models import RAGDocument
    
    # Defensive check: Ensure metadata is never None or 'null'
    # This provides a second layer of protection against NOT NULL constraint violations
    if metadata is None or metadata == 'null':
        metadata = '{}'
    
    # Check if document exists
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.doc_id == doc_id)
    )
    existing_doc = result.scalar_one_or_none()
    
    if existing_doc:
        # Update existing document
        existing_doc.content = content
        existing_doc.metadata = metadata
        existing_doc.user_id = user_id
        await db.commit()
        await db.refresh(existing_doc)
        return existing_doc
    else:
        # Create new document
        rag_doc = RAGDocument(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            user_id=user_id
        )
        db.add(rag_doc)
        await db.commit()
        await db.refresh(rag_doc)
        return rag_doc


async def get_rag_document(
    db: AsyncSession,
    doc_id: str
) -> Optional["RAGDocument"]:
    """Get a RAG document by ID."""
    from db.models import RAGDocument
    
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.doc_id == doc_id)
    )
    return result.scalar_one_or_none()


async def get_rag_documents_bulk(
    db: AsyncSession,
    doc_ids: List[str]
) -> List["RAGDocument"]:
    """Get multiple RAG documents by IDs."""
    from db.models import RAGDocument
    
    if not doc_ids:
        return []
    
    result = await db.execute(
        select(RAGDocument).where(RAGDocument.doc_id.in_(doc_ids))
    )
    return result.scalars().all()


async def delete_rag_document(
    db: AsyncSession,
    doc_id: str
) -> bool:
    """Delete a RAG document by ID."""
    from db.models import RAGDocument
    
    result = await db.execute(
        delete(RAGDocument).where(RAGDocument.doc_id == doc_id)
    )
    await db.commit()
    return result.rowcount > 0


async def clear_all_rag_documents(db: AsyncSession) -> int:
    """Clear all RAG documents from database."""
    from db.models import RAGDocument
    
    result = await db.execute(delete(RAGDocument))
    await db.commit()
    return result.rowcount
