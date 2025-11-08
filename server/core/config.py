"""
Configuration file for ECL calculation API.

Contains paths, constants, and settings for:
- Data directories
- ECL calculation parameters
- Pinecone database configuration
- Segment columns
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory (Interview/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present (no error if missing)
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Data directories (supports read-only deployments like Vercel)
data_root_env = os.getenv("DATA_ROOT")
if data_root_env:
    data_root_candidate = Path(data_root_env)
    DATA_DIR = data_root_candidate if data_root_candidate.is_absolute() else (BASE_DIR / data_root_candidate).resolve()
else:
    DATA_DIR = (BASE_DIR / "data").resolve()

try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Fall back to /tmp on platforms with read-only deployment directory
    DATA_DIR = Path("/tmp/risk-analysis-data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SEGMENTS_DIR = DATA_DIR / "segments"

for directory in (UPLOAD_DIR, PROCESSED_DIR, SEGMENTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ECL Calculation Constants
BASE_LGD = 0.35  # Base Loss Given Default
LGD_RATE_MULTIPLIER = 0.05  # Interest rate adjustment multiplier

# Segment columns to analyze
SEGMENT_COLUMNS = [
    "loan_intent",
    "person_gender",
    "person_education",
    "person_home_ownership",
    "age_group",  # Added age segmentation
]

# Risk thresholds for recommendations
PD_THRESHOLD_HIGH = 0.25  # High risk threshold for PD
PD_THRESHOLD_MEDIUM = 0.15  # Medium risk threshold for PD

# Minimum sample size for segment statistics (configurable via env var)
MIN_SEGMENT_SIZE = int(os.getenv("MIN_SEGMENT_SIZE", "1"))

# RBAC Configuration
# Auto-grant default read permissions to Analysts for core segments
RBAC_AUTO_GRANT_DEFAULTS = os.getenv("RBAC_AUTO_GRANT_DEFAULTS", "true").lower() == "true"
# Enable fuzzy matching for segment permissions (handles case/spelling variations)
RBAC_FUZZY_MATCHING = os.getenv("RBAC_FUZZY_MATCHING", "true").lower() == "true"

_auto_grant_env = os.getenv("RBAC_AUTO_GRANT_SEGMENTS")
if _auto_grant_env:
    RBAC_AUTO_GRANT_SEGMENTS = {
        value.strip()
        for value in _auto_grant_env.split(",")
        if value.strip()
    }
else:
    RBAC_AUTO_GRANT_SEGMENTS = {
        "loan_intent",
        "person_gender",
        "person_education",
        "person_home_ownership",
    }

# Pinecone Configuration
# Note: Pinecone serverless (2025-04 API) does not support the embed.model parameter.
# Embeddings are generated client-side using OpenAI and upserted as raw vectors.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "developer-quickstart-py")
PINECONE_DIMENSION = 1536  # Must match OpenAI text-embedding-3-small dimension
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # For recommendations and RAG
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Validation: Ensure dimension alignment
if PINECONE_DIMENSION != 1536:
    raise ValueError(
        f"PINECONE_DIMENSION ({PINECONE_DIMENSION}) must be 1536 to match "
        f"{OPENAI_EMBEDDING_MODEL} embedding dimension"
    )

# RAG Configuration
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # Number of segments to retrieve
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))  # Chunk size for splitting
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))  # Overlap between chunks

# IQR outlier capping columns (from preprocess.py)
IQR_COLS = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

# Expected columns in uploaded data
REQUIRED_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "loan_status"
]

# Age group boundaries for segmentation
AGE_GROUPS = {
    "young": (0, 35),
    "middle_aged": (35, 55),
    "senior_citizen": (55, 100)
}

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Neon PostgreSQL connection string
if not DATABASE_URL:
    raise ValueError("[ERROR] DATABASE_URL environment variable must be set")

# JWT Authentication Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("[ERROR] JWT_SECRET_KEY environment variable must be set. Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

