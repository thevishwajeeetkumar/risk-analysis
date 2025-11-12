"""
SQLAlchemy models that map to existing Neon database tables.

These models represent the schema already created in the Neon database.
No table creation will occur - these only map to existing tables.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """Maps to existing 'users' table in Neon database."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    email = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'Analyst' or 'CRO'
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    loans = relationship("Loan", back_populates="user", lazy="noload")
    ecl_calculations = relationship("ECLSegmentCalculation", back_populates="user", lazy="noload")
    permissions = relationship("Permission", back_populates="user", lazy="noload")


class Loan(Base):
    """Maps to existing 'loans' table in Neon database."""
    __tablename__ = "loans"
    __table_args__ = {'extend_existing': True}
    
    loan_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    
    # Loan details
    loan_amount = Column(Float, nullable=False)
    loan_intent = Column(String, nullable=False)
    loan_int_rate = Column(Float, nullable=False)
    loan_percent_income = Column(Float, nullable=False)
    
    # Borrower details
    credit_score = Column(Integer, nullable=False)
    person_income = Column(Float, nullable=False)
    person_age = Column(Integer, nullable=False)
    person_gender = Column(String, nullable=False)
    person_education = Column(String, nullable=False)
    person_emp_exp = Column(Integer, nullable=True)  # Employment experience
    home_ownership = Column(String, nullable=False)  # RENT, OWN, MORTGAGE, OTHER
    
    # Credit history
    cb_person_cred_hist_length = Column(Integer, nullable=False)
    previous_loan_defaults_on_file = Column(Boolean, nullable=False, server_default=text("false"))
    
    # Loan status
    loan_status = Column(Integer, nullable=False)  # 1 = paid, 0 = defaulted
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="loans", lazy="noload")
    ecl_calculations = relationship("ECLSegmentCalculation", back_populates="loan", lazy="noload")


class ECLSegmentCalculation(Base):
    """Maps to existing 'ecl_segment_calculation' table in Neon database."""
    __tablename__ = "ecl_segment_calculation"
    __table_args__ = {'extend_existing': True}
    
    ecl_id = Column(Integer, primary_key=True, autoincrement=True)
    loan_id = Column(Integer, ForeignKey("loans.loan_id"), nullable=True)  # NULL for aggregate calculations
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    
    # Segment information
    segment_name = Column(String, nullable=False)  # e.g., "loan_intent", "age_group"
    segment_value = Column(String, nullable=False)  # e.g., "PERSONAL", "senior_citizen"
    
    # ECL metrics
    pd_value = Column(Float, nullable=False)  # Probability of Default
    lgd_value = Column(Float, nullable=False)  # Loss Given Default
    ead_value = Column(Float, nullable=False)  # Exposure at Default
    ecl_value = Column(Float, nullable=False)  # Expected Credit Loss
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="ecl_calculations", lazy="noload")
    loan = relationship("Loan", back_populates="ecl_calculations", lazy="noload")


class Permission(Base):
    """Maps to existing 'permissions' table in Neon database for RBAC."""
    __tablename__ = "permissions"
    __table_args__ = {'extend_existing': True}
    
    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    segment_name = Column(String, nullable=False)  # e.g., "loan_intent", "age_group"
    permission_type = Column(String, nullable=False)  # 'read' or 'write'
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="permissions", lazy="noload")
    
    # Composite unique constraint (user can't have duplicate permissions for same segment)
    __table_args__ = (
        {'extend_existing': True},
    )


class RAGDocument(Base):
    """Persistent storage for RAG parent documents."""
    __tablename__ = "rag_documents"
    __table_args__ = {'extend_existing': True}
    
    doc_id = Column(String, primary_key=True)  # Document identifier (parent_id)
    content = Column(Text, nullable=False)  # Document content
    doc_metadata = Column("metadata", Text, nullable=False)  # JSON-encoded metadata (renamed to avoid SQLAlchemy conflict)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
