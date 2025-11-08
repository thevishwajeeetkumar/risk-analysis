"""
FastAPI Application for ECL Calculation and RAG Querying

Endpoints:
1. POST /api/upload - Upload CSV/XLSX, process pipeline, embed to Pinecone
2. POST /api/query - Natural language query on ECL segments
3. GET /api/segments - View processed segment data
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.pipeline import process_uploaded_file
from core.rag_engine import get_rag_engine
from core.recommendation import generate_query_verdict
from core.segmentation import load_segment_results
from core.config import UPLOAD_DIR, SEGMENTS_DIR

# Database and auth imports
from db.database import get_db, async_session_factory
from db import crud
from db.models import User
from auth.auth import (
    authenticate_user,
    create_user_token,
    get_current_user,
    get_current_cro_user,
    hash_password
)
from auth.schemas import (
    UserCreate,
    UserResponse,
    Token,
    PermissionGrant,
    UserPermissions,
    PermissionCreate
)
from auth.rbac import filter_segments_dict_by_permissions


# Initialize logger
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title="ECL Calculation & RAG API",
    description="Direct ECL calculation with natural language querying using RAG",
    version="1.0.0"
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - configured from environment variable
# Default allows localhost for development
# Production should set ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://risk-frontend-snowy.vercel.app",
]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", ",".join(default_origins)).split(",")
print(f"[INFO] CORS configured for origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_schema_checks():
    """Ensure database schema is compatible before serving requests."""
    from db.schema_check import ensure_loans_schema, ensure_ecl_schema

    async with async_session_factory() as session:
        await ensure_loans_schema(session)
        await ensure_ecl_schema(session)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    file_id: Optional[str] = None
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    query: str
    verdict: str
    rag_answer: str
    recommendations: List[dict]
    summary: str
    metrics: dict


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ECL Calculation & RAG API",
        "version": "1.0.0",
        "endpoints": {
            "auth": {
                "register": "POST /auth/register - Register new user (open to all)",
                "login": "POST /auth/login - Login and get JWT token",
                "me": "GET /auth/me - Get current user info",
                "grant_permissions": "POST /auth/permissions - Grant permissions (CRO only)",
                "view_permissions": "GET /auth/permissions/{user_id} - View user permissions"
            },
            "ecl": {
                "upload": "POST /api/upload - Upload and process loan data",
                "query": "POST /api/query - Query ECL segments with natural language",
                "segments": "GET /api/segments - View processed segment data"
            }
        },
        "status": "online"
    }


@app.post("/api/upload")
@limiter.limit("5/minute")  # Limit to 5 uploads per minute per user
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload CSV/XLSX file and run complete ECL pipeline (requires authentication).
    
    Pipeline:
    1. Clean data (preprocess)
    2. Create age groups from person_age column
    3. Store individual loans in database
    4. Segment by loan_intent, gender, education, home_ownership, age_group
    5. Calculate ECL (PD × LGD × EAD)
    6. Store ECL calculations in database
    7. Save segment CSVs (backward compatibility)
    8. Embed to Pinecone for RAG querying
    
    Returns:
        - file_id: Unique identifier for this upload
        - status: "success" or "error"
        - loan_count: Number of loans stored
        - statistics: Summary metrics (total_loans, avg_pd, total_ecl, etc.)
        - segments: List of segment types processed
    
    Requires:
        - Valid JWT token in Authorization header
    """
    # Validate file type
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload CSV or XLSX file. Note: PDF support is not available in this version."
        )
    
    # Validate file size (limit to 100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 100MB. Your file is {file_size / (1024*1024):.2f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty. Please upload a valid file."
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_path = Path(UPLOAD_DIR) / f"{file_id}_{file.filename}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Contents already read above for size validation
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        print(f"\n[FILE] Uploaded: {upload_path}")
        
        # Run async pipeline with database storage
        result = await process_uploaded_file(
            file_path=str(upload_path),
            file_id=file_id,
            user_id=current_user.user_id,
            username=current_user.username,
            db=db
        )
        
        return JSONResponse(content=result, status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/api/query")
@limiter.limit("30/minute")  # Limit to 30 queries per minute per user
async def query_ecl(
    request: Request,
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Query ECL segments using natural language (requires authentication).
    
    Uses RAG (Retrieval-Augmented Generation) with:
    - Pinecone vector database
    - OpenAI embeddings
    - LangChain simple vector retrieval
    - RBAC filtering based on user permissions
    
    Args:
        query: Natural language question about ECL segments
        file_id: Optional filter by specific upload
        top_k: Number of segments to retrieve (default 5)
    
    Returns:
        - query: Original query
        - verdict: Overall risk assessment (HIGH/MEDIUM/LOW)
        - rag_answer: Generated answer from LLM
        - recommendations: List of actionable recommendations (filtered by permissions)
        - summary: Summary of findings
        - metrics: Statistics (avg_pd, total_ecl, etc.)
    
    Requires:
        - Valid JWT token in Authorization header
        - Read permissions for relevant segments
    """
    try:
        # Get RAG engine
        rag_engine = get_rag_engine()
        
        # Query segments
        rag_result = rag_engine.query_segments(
            query=payload.query,
            username=current_user.username,
            file_id=payload.file_id,
            top_k=payload.top_k
        )
        
        # Log RAG retrieval results
        rag_segment_types = set(s.get('segment_type') for s in rag_result['segments'] if s.get('segment_type'))
        logger.info(f"[RAG] Retrieved {len(rag_result['segments'])} segments, types: {rag_segment_types}")
        logger.info(f"[USER] {current_user.username} (role: {current_user.role}, ID: {current_user.user_id}) querying file: {payload.file_id or 'all'}")
        
        # Apply RBAC filtering to segments
        # Group segments by their segment type for filtering
        segments_by_type = {}
        for segment in rag_result["segments"]:
            seg_type = segment.get("segment_type")
            if seg_type:
                if seg_type not in segments_by_type:
                    segments_by_type[seg_type] = []
                segments_by_type[seg_type].append(segment)
        
        logger.info(f"[RBAC] Pre-filter: {len(segments_by_type)} segment types with {len(rag_result['segments'])} total segments")
        
        # Filter segments based on user permissions
        filtered_segments_dict = await filter_segments_dict_by_permissions(
            current_user,
            segments_by_type,
            db
        )
        
        # Log RBAC filtering results
        retained_types = list(filtered_segments_dict.keys())
        removed_types = set(segments_by_type.keys()) - set(retained_types)
        logger.info(f"[RBAC] Post-filter: {len(filtered_segments_dict)} types retained {retained_types[:3]}{'...' if len(retained_types) > 3 else ''}")
        if removed_types:
            logger.warning(f"[RBAC] Filtered out types: {removed_types}")
        
        # Flatten filtered segments back to list
        filtered_segments = []
        for seg_list in filtered_segments_dict.values():
            filtered_segments.extend(seg_list)
        
        # Warn if all segments were filtered out
        segments_culled = len(rag_result['segments']) - len(filtered_segments)
        if not filtered_segments:
            logger.warning(f"[RBAC] All segments filtered! RBAC culled {segments_culled} segments. User may lack permissions.")
        else:
            logger.info(f"[RBAC] Final: {len(filtered_segments)} segments (culled {segments_culled})")
        
        # Generate verdict with filtered recommendations
        verdict_result = generate_query_verdict(
            query=payload.query,
            segments=filtered_segments,
            rag_answer=rag_result["answer"]
        )
        
        return JSONResponse(content=verdict_result, status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying segments: {str(e)}"
        )


@app.get("/api/segments")
async def get_segments(
    file_id: Optional[str] = Query(None, description="Filter by file ID"),
    segment_type: Optional[str] = Query(
        None,
        description="Filter by segment type (loan_intent, person_gender, person_education, person_home_ownership, age_group)"
    ),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve processed ECL segment data from database (requires authentication).
    
    Args:
        file_id: Optional filter by specific upload
        segment_type: Optional filter by segment type
        skip: Number of items to skip for pagination
        limit: Number of items to return (max 1000)
    
    Returns:
        - segments: List of ECL calculations (filtered by RBAC permissions)
        - total: Total number of segments (before pagination)
        - skip: Number skipped
        - limit: Limit used
    
    Requires:
        - Valid JWT token in Authorization header
        - Read permissions for requested segments
    """
    try:
        from auth.rbac import filter_ecl_by_permissions
        from db.transformers import ecl_record_to_segment_dict
        
        # Query ECL calculations from database
        ecl_records = await crud.get_ecl_by_user(
            db,
            user_id=current_user.user_id,
            file_id=file_id,
            segment_type=segment_type,
            skip=skip,
            limit=limit
        )
        
        # Apply RBAC filtering
        filtered_ecl_records = await filter_ecl_by_permissions(
            current_user,
            ecl_records,
            db
        )
        
        # Convert to response format
        segments_data = []
        for ecl_record in filtered_ecl_records:
            segment_dict = ecl_record_to_segment_dict(ecl_record)
            segments_data.append(segment_dict)
        
        # Get total count (before pagination)
        # For now, we'll use the filtered count as total
        total_count = len(segments_data)
        
        # Group by segment type if no specific type requested
        if not segment_type:
            grouped_data = {}
            for segment in segments_data:
                seg_type = segment["segment_type"]
                if seg_type not in grouped_data:
                    grouped_data[seg_type] = []
                grouped_data[seg_type].append(segment)
            
            return JSONResponse(content={
                "message": f"Found {total_count} ECL calculations across {len(grouped_data)} segment types",
                "segments_by_type": grouped_data,
                "total": total_count,
                "skip": skip,
                "limit": limit
            }, status_code=200)
        else:
            # Return flat list for specific segment type
            return JSONResponse(content={
                "message": f"Found {len(segments_data)} ECL calculations for {segment_type}",
                "segments": segments_data,
                "total": total_count,
                "skip": skip,
                "limit": limit
            }, status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving segments: {str(e)}"
        )


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/auth/register", response_model=UserResponse)
@limiter.limit("3/hour")  # Limit to 3 registrations per hour per IP
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.
    
    Anyone can register. Users select their role during signup.
    This endpoint does NOT require authentication.
    """
    print(f"\n[INFO] Registration request received:")
    print(f"   Username: {user_data.username}")
    print(f"   Email: {user_data.email}")
    print(f"   Role: {user_data.role}")
    
    try:
        # Validate input data
        if not user_data.username or len(user_data.username.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username is required"
            )
        
        if not user_data.email or len(user_data.email.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        if not user_data.password or len(user_data.password) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is required"
            )
        
        if user_data.role not in ['Analyst', 'CRO']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be either 'Analyst' or 'CRO'"
            )
        
        # Check if username already exists
        print("[CHECK] Checking if username exists...")
        existing_user = await crud.get_user_by_username(db, user_data.username)
        if existing_user:
            print(f"[ERROR] Username '{user_data.username}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        print("[CHECK] Checking if email exists...")
        existing_email = await crud.get_user_by_email(db, user_data.email)
        if existing_email:
            print(f"[ERROR] Email '{user_data.email}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        print("[LOCK] Hashing password...")
        hashed_password = hash_password(user_data.password)
        
        # Create user
        print("[SAVE] Creating user in database...")
        user = await crud.create_user(
            db,
            username=user_data.username.strip(),
            password_hash=hashed_password,
            email=user_data.email.strip().lower(),
            role=user_data.role
        )
        
        print(f"[SUCCESS] User created in database: {user.username} (ID: {user.user_id})")
        
        # Convert to response format
        try:
            # Try Pydantic v2 first
            user_response = UserResponse.model_validate(user)
        except AttributeError:
            # Fallback to Pydantic v1
            user_response = UserResponse.from_orm(user)
        except Exception as validation_error:
            print(f"[WARNING]  Pydantic validation error, using manual construction: {str(validation_error)}")
            # Manual construction as fallback
            user_response = UserResponse(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at
            )
        
        print(f"[SUCCESS] Registration successful: {user_response.username} (ID: {user_response.user_id})")
        return user_response
            
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as-is
        print(f"[ERROR] HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Log the full error for debugging
        print(f"\n[ERROR] ERROR during user registration:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        print("   Traceback:")
        traceback.print_exc()
        
        # Return a user-friendly error message
        error_detail = str(e)
        if "duplicate key" in error_detail.lower() or "unique constraint" in error_detail.lower():
            if "username" in error_detail.lower():
                error_detail = "Username already registered"
            elif "email" in error_detail.lower():
                error_detail = "Email already registered"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {error_detail}"
        )


@app.post("/auth/login", response_model=Token)
@limiter.limit("10/minute")  # Limit to 10 login attempts per minute per IP
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint that returns JWT access token.
    
    Use the token in Authorization header as: Bearer <token>
    """
    print(f"\n[INFO] Login request received:")
    print(f"   Username: {form_data.username}")
    
    try:
        user = await authenticate_user(db, form_data.username, form_data.password)
        if not user:
            print(f"[ERROR] Authentication failed for username: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        print(f"[SUCCESS] User authenticated: {user.username} (ID: {user.user_id}, Role: {user.role})")
        access_token = create_user_token(user)
        
        print(f"[SUCCESS] Token created successfully")
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=user.user_id,
            username=user.username,
            role=user.role
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error during login: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    # Use model_validate for Pydantic v2, fallback to from_orm for v1
    try:
        return UserResponse.model_validate(current_user)
    except AttributeError:
        # Pydantic v1 compatibility
        return UserResponse.from_orm(current_user)


@app.get("/admin/users", response_model=List[UserResponse])
async def get_all_users(
    current_user: User = Depends(get_current_cro_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all users (CRO only).
    
    Returns a list of all registered users.
    """
    try:
        users = await crud.list_users(db)
        # Use model_validate for Pydantic v2, fallback to from_orm for v1
        result = []
        for user in users:
            try:
                result.append(UserResponse.model_validate(user))
            except AttributeError:
                result.append(UserResponse.from_orm(user))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving users: {str(e)}"
        )


@app.post("/auth/permissions", response_model=List[PermissionCreate])
async def grant_permissions(
    permission_grant: PermissionGrant,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_cro_user)  # Only CROs can grant permissions
):
    """
    Grant permissions to a user (CRO only endpoint).
    
    Allows CROs to grant segment-level read/write permissions to Analysts.
    """
    # Check if target user exists
    target_user = await crud.get_user_by_id(db, permission_grant.user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Convert permission data to dicts
    permissions_data = [
        {
            "segment_name": perm.segment_name,
            "permission_type": perm.permission_type
        }
        for perm in permission_grant.permissions
    ]
    
    # Grant permissions
    created_permissions = await crud.grant_permissions(
        db,
        permission_grant.user_id,
        permissions_data
    )
    
    return permission_grant.permissions


@app.get("/auth/permissions/{user_id}", response_model=UserPermissions)
async def get_user_permissions(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get permissions for a specific user.
    
    - CROs can view any user's permissions
    - Analysts can only view their own permissions
    """
    # Check authorization
    if current_user.role != "CRO" and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's permissions"
        )
    
    # Get user and permissions
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    permissions = await crud.get_user_permissions(db, user_id)
    
    return UserPermissions(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        permissions=permissions
    )


# ==================== DATABASE QUERY ENDPOINTS ====================

@app.get("/api/loans")
async def get_loans(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    loan_intent: Optional[str] = Query(None, description="Filter by loan intent"),
    loan_status: Optional[int] = Query(None, description="Filter by loan status (0=defaulted, 1=paid)"),
    min_amount: Optional[float] = Query(None, description="Minimum loan amount"),
    max_amount: Optional[float] = Query(None, description="Maximum loan amount"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get loans with pagination and filters (requires authentication).
    
    Returns:
        - loans: List of loan records
        - total: Total number of loans
        - skip: Number skipped
        - limit: Limit used
    """
    try:
        # Get loans for current user
        loans = await crud.get_loans_with_filters(
            db,
            user_id=current_user.user_id,
            loan_intent=loan_intent,
            loan_status=loan_status,
            min_amount=min_amount,
            max_amount=max_amount,
            skip=skip,
            limit=limit
        )
        
        # Get total count
        total_count = await crud.count_loans_by_user(db, current_user.user_id)
        
        # Convert to response format
        loans_data = []
        for loan in loans:
            loans_data.append({
                "loan_id": loan.loan_id,
                "loan_amount": loan.loan_amount,
                "loan_intent": loan.loan_intent,
                "loan_int_rate": loan.loan_int_rate,
                "credit_score": loan.credit_score,
                "person_age": loan.person_age,
                "loan_status": loan.loan_status,
                "created_at": loan.created_at.isoformat()
            })
        
        return {
            "loans": loans_data,
            "total": total_count,
            "skip": skip,
            "limit": limit
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving loans: {str(e)}"
        )


@app.get("/api/loans/{loan_id}")
async def get_loan_detail(
    loan_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific loan (requires authentication).
    
    Args:
        loan_id: ID of the loan to retrieve
    
    Returns:
        Complete loan information
    """
    try:
        loan = await crud.get_loan_by_id(db, loan_id)
        
        if not loan:
            raise HTTPException(
                status_code=404,
                detail="Loan not found"
            )
        
        # Check if user owns the loan
        if loan.user_id != current_user.user_id and current_user.role != "CRO":
            raise HTTPException(
                status_code=403,
                detail="Not authorized to view this loan"
            )
        
        return {
            "loan_id": loan.loan_id,
            "user_id": loan.user_id,
            "loan_amount": loan.loan_amount,
            "loan_intent": loan.loan_intent,
            "loan_int_rate": loan.loan_int_rate,
            "loan_percent_income": loan.loan_percent_income,
            "credit_score": loan.credit_score,
            "person_income": loan.person_income,
            "person_age": loan.person_age,
            "person_gender": loan.person_gender,
            "person_education": loan.person_education,
            "person_emp_exp": loan.person_emp_exp,
            "home_ownership": loan.home_ownership,
            "cb_person_cred_hist_length": loan.cb_person_cred_hist_length,
            "previous_loan_defaults_on_file": loan.previous_loan_defaults_on_file,
            "loan_status": loan.loan_status,
            "created_at": loan.created_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving loan: {str(e)}"
        )


@app.get("/api/ecl/history")
async def get_ecl_history(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    segment_name: Optional[str] = Query(None, description="Filter by segment type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get ECL calculation history for the last N days (requires authentication).
    
    Args:
        days: Number of days to look back (default 30, max 365)
        segment_name: Optional filter by segment type
    
    Returns:
        List of ECL calculations with timestamps
    """
    try:
        from auth.rbac import filter_ecl_by_permissions
        from db.transformers import ecl_record_to_segment_dict
        
        # Get ECL history
        ecl_history = await crud.get_ecl_history(
            db,
            user_id=current_user.user_id,
            days=days,
            segment_name=segment_name
        )
        
        # Apply RBAC filtering
        filtered_history = await filter_ecl_by_permissions(
            current_user,
            ecl_history,
            db
        )
        
        # Group by date and segment type
        history_by_date = {}
        for ecl_record in filtered_history:
            date_key = ecl_record.created_at.date().isoformat()
            if date_key not in history_by_date:
                history_by_date[date_key] = []
            
            segment_dict = ecl_record_to_segment_dict(ecl_record)
            history_by_date[date_key].append(segment_dict)
        
        return {
            "history": history_by_date,
            "total_calculations": len(filtered_history),
            "days_requested": days,
            "segment_filter": segment_name
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving ECL history: {str(e)}"
        )


@app.get("/api/users/{user_id}/accessible-segments")
async def get_accessible_segments(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of segments a user has access to (requires authentication).
    
    Args:
        user_id: ID of user to check permissions for
    
    Returns:
        List of accessible segment names
    
    Note:
        - CROs can view any user's accessible segments
        - Analysts can only view their own
    """
    try:
        # Check authorization
        if current_user.role != "CRO" and current_user.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to view this user's accessible segments"
            )
        
        # Get user to verify they exist
        target_user = await crud.get_user_by_id(db, user_id)
        if not target_user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        from auth.rbac import get_user_accessible_segments
        
        # Get accessible segments
        accessible_segments = await get_user_accessible_segments(target_user, db)
        
        return {
            "user_id": user_id,
            "username": target_user.username,
            "role": target_user.role,
            "accessible_segments": accessible_segments,
            "total_segments": len(accessible_segments)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving accessible segments: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ECL RAG API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

