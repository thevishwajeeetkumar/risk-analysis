"""
Authentication utilities for password hashing and JWT token management.

Provides secure password hashing with bcrypt and JWT token creation/validation.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db import crud
from db.models import User
from core.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# OAuth2 scheme for token extraction
# auto_error=False allows endpoints to handle missing tokens gracefully
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=True)


# ==================== Password Hashing ====================

def hash_password(plain_password: str) -> str:
    """
    Hash a plain password using bcrypt.
    
    Args:
        plain_password: Plain text password
        
    Returns:
        Hashed password string (bcrypt hash)
    """
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to verify against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        if not plain_password or not hashed_password:
            print("[WARNING]  Empty password or hash provided for verification")
            return False
        
        if not hashed_password.startswith('$2b$') and not hashed_password.startswith('$2a$') and not hashed_password.startswith('$2y$'):
            print(f"[WARNING]  Invalid hash format (doesn't start with $2b$, $2a$, or $2y$): {hashed_password[:20]}...")
            return False
        
        result = bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
        if not result:
            print("[WARNING]  Password hash mismatch during verification")
        return result
    except Exception as e:
        print(f"[ERROR] Error verifying password: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==================== JWT Token Management ====================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Create token
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token data
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== User Authentication ====================

async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str
) -> Optional[User]:
    """
    Authenticate a user by username and password.
    
    Args:
        db: Database session
        username: Username to authenticate
        password: Plain text password
        
    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        print(f"[CHECK] Authenticating user: {username}")
        user = await crud.get_user_by_username(db, username)
        if not user:
            print(f"[ERROR] User not found: {username}")
            return None
        
        print(f"[SUCCESS] User found: {user.username} (ID: {user.user_id})")
        password_valid = verify_password(password, user.password_hash)
        if not password_valid:
            print(f"[ERROR] Password verification failed for user: {username}")
            return None
        
        print(f"[SUCCESS] Password verified successfully")
        return user
    except Exception as e:
        print(f"[ERROR] Error during authentication: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== FastAPI Dependencies ====================

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    FastAPI dependency that extracts and validates the JWT token,
    then retrieves the corresponding user from the database.
    
    Args:
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        Current authenticated User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token
        payload = decode_access_token(token)
        user_id_str = payload.get("sub")
        
        if user_id_str is None:
            raise credentials_exception
        
        # Convert user_id from string to int (JWT stores it as string)
        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            print(f"[ERROR] Invalid user_id in token: {user_id_str} (type: {type(user_id_str)})")
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error decoding token or getting user: {str(e)}")
        raise credentials_exception
    
    # Get user from database
    try:
        user = await crud.get_user_by_id(db, user_id=user_id)
        if user is None:
            print(f"[ERROR] User not found with ID: {user_id}")
            raise credentials_exception
        
        print(f"[SUCCESS] Current user retrieved: {user.username} (ID: {user.user_id})")
        return user
    except Exception as e:
        print(f"[ERROR] Database error getting user: {str(e)}")
        raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Verify that the current user is active.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        Current user if active
        
    Raises:
        HTTPException: If user is inactive
    """
    # In this implementation, we don't have an 'active' field,
    # but this can be extended to check user status
    return current_user


async def get_current_cro_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Verify that the current user has CRO role.
    
    FastAPI dependency for endpoints that require CRO access.
    
    Args:
        current_user: User from get_current_active_user dependency
        
    Returns:
        Current user if they have CRO role
        
    Raises:
        HTTPException: If user is not a CRO
    """
    if current_user.role != "CRO":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. CRO role required."
        )
    return current_user


# ==================== Token Creation Helpers ====================

def create_user_token(user: User) -> str:
    """
    Create an access token for a user.
    
    Note: JWT standard requires string values, so user_id is stored as string.
    It will be converted back to int in get_current_user().
    
    Args:
        user: User object to create token for
        
    Returns:
        JWT access token string
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Store user_id as string in JWT (JWT standard), will be converted to int when decoding
    access_token = create_access_token(
        data={"sub": str(user.user_id), "role": user.role},
        expires_delta=access_token_expires
    )
    print(f"[SUCCESS] Token created for user_id: {user.user_id} (stored as string in JWT)")
    return access_token

