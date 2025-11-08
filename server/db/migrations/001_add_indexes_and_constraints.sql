-- Migration: Add Performance Indexes and Constraints
-- Date: November 7, 2025
-- Description: Adds indexes for query performance and constraints for data integrity

-- ============================================================
-- PART 1: CREATE TABLE FOR RAG DOCUMENTS (if not exists)
-- ============================================================

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id VARCHAR PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

COMMENT ON TABLE rag_documents IS 'Persistent storage for RAG parent documents';

-- Ensure metadata has a safe default to avoid NULL insertions
ALTER TABLE rag_documents
    ALTER COLUMN metadata SET DEFAULT '{}';

-- ============================================================
-- PART 2: ADD PERFORMANCE INDEXES
-- ============================================================

-- Indexes on ECL Segment Calculation table
CREATE INDEX IF NOT EXISTS idx_ecl_segment_name ON ecl_segment_calculation(segment_name);
CREATE INDEX IF NOT EXISTS idx_ecl_user_id ON ecl_segment_calculation(user_id);
CREATE INDEX IF NOT EXISTS idx_ecl_created_at ON ecl_segment_calculation(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ecl_segment_value ON ecl_segment_calculation(segment_value);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_ecl_user_segment ON ecl_segment_calculation(user_id, segment_name, created_at DESC);

-- Indexes on Loans table
CREATE INDEX IF NOT EXISTS idx_loans_user_id ON loans(user_id);
CREATE INDEX IF NOT EXISTS idx_loans_loan_intent ON loans(loan_intent);
CREATE INDEX IF NOT EXISTS idx_loans_created_at ON loans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_loans_loan_status ON loans(loan_status);

-- Composite index for filtering
CREATE INDEX IF NOT EXISTS idx_loans_user_status ON loans(user_id, loan_status);

-- Indexes on Permissions table
CREATE INDEX IF NOT EXISTS idx_permissions_user_id ON permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_permissions_segment_name ON permissions(segment_name);

-- Index on RAG documents
CREATE INDEX IF NOT EXISTS idx_rag_docs_user_id ON rag_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_rag_docs_created_at ON rag_documents(created_at DESC);

-- Index on Users table
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ============================================================
-- PART 3: ADD UNIQUE CONSTRAINTS
-- ============================================================

-- Unique constraint on permissions (user cannot have duplicate permissions)
-- First, remove duplicates if they exist
DO $$
BEGIN
    -- Delete duplicate permissions, keeping the oldest one
    DELETE FROM permissions a USING (
        SELECT MIN(permission_id) as permission_id, user_id, segment_name, permission_type
        FROM permissions
        GROUP BY user_id, segment_name, permission_type
        HAVING COUNT(*) > 1
    ) b
    WHERE a.user_id = b.user_id 
      AND a.segment_name = b.segment_name 
      AND a.permission_type = b.permission_type
      AND a.permission_id <> b.permission_id;
END $$;

-- Now add the unique constraint
ALTER TABLE permissions DROP CONSTRAINT IF EXISTS unique_user_segment_permission;
ALTER TABLE permissions ADD CONSTRAINT unique_user_segment_permission 
    UNIQUE (user_id, segment_name, permission_type);

-- ============================================================
-- PART 4: ADD CHECK CONSTRAINTS (if needed)
-- ============================================================

-- Ensure permission_type is valid
ALTER TABLE permissions DROP CONSTRAINT IF EXISTS check_permission_type;
ALTER TABLE permissions ADD CONSTRAINT check_permission_type 
    CHECK (permission_type IN ('read', 'write'));

-- Ensure user role is valid
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_user_role;
ALTER TABLE users ADD CONSTRAINT check_user_role 
    CHECK (role IN ('Analyst', 'CRO'));

-- Ensure loan_status is valid (0 = defaulted, 1 = paid)
ALTER TABLE loans DROP CONSTRAINT IF EXISTS check_loan_status;
ALTER TABLE loans ADD CONSTRAINT check_loan_status 
    CHECK (loan_status IN (0, 1));

-- ============================================================
-- PART 5: ADD FOREIGN KEY INDEXES (if not already indexed)
-- ============================================================

-- These are already covered above, but listed for completeness:
-- - idx_ecl_user_id (foreign key to users)
-- - idx_loans_user_id (foreign key to users)
-- - idx_permissions_user_id (foreign key to users)
-- - idx_rag_docs_user_id (foreign key to users)

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check all indexes
DO $$
BEGIN
    RAISE NOTICE 'Indexes created successfully!';
    RAISE NOTICE 'To verify indexes, run: SELECT tablename, indexname FROM pg_indexes WHERE schemaname = ''public'' ORDER BY tablename, indexname;';
END $$;

-- ============================================================
-- ROLLBACK SCRIPT (for reference, do not execute)
-- ============================================================

/*
-- To rollback this migration, run:

DROP INDEX IF EXISTS idx_ecl_segment_name;
DROP INDEX IF EXISTS idx_ecl_user_id;
DROP INDEX IF EXISTS idx_ecl_created_at;
DROP INDEX IF EXISTS idx_ecl_segment_value;
DROP INDEX IF EXISTS idx_ecl_user_segment;
DROP INDEX IF EXISTS idx_loans_user_id;
DROP INDEX IF EXISTS idx_loans_loan_intent;
DROP INDEX IF EXISTS idx_loans_created_at;
DROP INDEX IF EXISTS idx_loans_loan_status;
DROP INDEX IF EXISTS idx_loans_user_status;
DROP INDEX IF EXISTS idx_permissions_user_id;
DROP INDEX IF EXISTS idx_permissions_segment_name;
DROP INDEX IF EXISTS idx_rag_docs_user_id;
DROP INDEX IF EXISTS idx_rag_docs_created_at;
DROP INDEX IF EXISTS idx_users_role;
DROP INDEX IF EXISTS idx_users_email;

ALTER TABLE permissions DROP CONSTRAINT IF EXISTS unique_user_segment_permission;
ALTER TABLE permissions DROP CONSTRAINT IF EXISTS check_permission_type;
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_user_role;
ALTER TABLE loans DROP CONSTRAINT IF EXISTS check_loan_status;
*/

