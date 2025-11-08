"""
Pydantic schemas for authentication requests and responses.

These models define the structure of data sent to and from auth endpoints.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=100)
    role: str = Field("Analyst", pattern="^(Analyst|CRO)$")
    
    @validator('password')
    def validate_password(cls, v):
        """Ensure password has at least one digit and one letter."""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isalpha() for char in v):
            raise ValueError('Password must contain at least one letter')
        return v


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class UserResponse(UserBase):
    """Schema for user responses (excludes password)."""
    user_id: int
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserList(BaseModel):
    """Schema for listing users."""
    users: List[UserResponse]
    total: int
    skip: int
    limit: int


# ==================== Token Schemas ====================

class Token(BaseModel):
    """Schema for authentication token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str


class TokenData(BaseModel):
    """Schema for token payload data."""
    user_id: Optional[int] = None
    role: Optional[str] = None


# ==================== Permission Schemas ====================

class PermissionBase(BaseModel):
    """Base permission schema."""
    segment_name: str = Field(..., min_length=1, max_length=100)
    permission_type: str = Field("read", pattern="^(read|write)$")


class PermissionCreate(PermissionBase):
    """Schema for creating a permission."""
    pass


class PermissionGrant(BaseModel):
    """Schema for granting permissions to a user."""
    user_id: int
    permissions: List[PermissionCreate]


class PermissionResponse(PermissionBase):
    """Schema for permission responses."""
    permission_id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserPermissions(BaseModel):
    """Schema for user's permissions."""
    user_id: int
    username: str
    role: str
    permissions: List[PermissionResponse]


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str
    status_code: int = 400
    

class ValidationError(BaseModel):
    """Validation error details."""
    loc: List[str]
    msg: str
    type: str


# ==================== Auth Status Schemas ====================

class AuthStatus(BaseModel):
    """Schema for authentication status."""
    is_authenticated: bool
    user: Optional[UserResponse] = None
    

class PasswordChange(BaseModel):
    """Schema for password change request."""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_new_password(cls, v, values):
        """Ensure new password is different and meets requirements."""
        if 'old_password' in values and v == values['old_password']:
            raise ValueError('New password must be different from old password')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isalpha() for char in v):
            raise ValueError('Password must contain at least one letter')
        return v

