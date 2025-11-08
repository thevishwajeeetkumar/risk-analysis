-- Migration: Drop CHECK Constraints from ecl_segment_calculation
-- Date: November 7, 2025
-- Description: Removes restrictive CHECK constraints to allow dynamic segment types

-- ============================================================
-- PART 1: DROP CHECK CONSTRAINTS
-- ============================================================

-- Drop CHECK constraint on segment_name (restricts allowed segment types)
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_segment_name_check;

-- Drop any CHECK constraint on segment_value
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_segment_value_check;

-- Drop any CHECK constraint on pd_value
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_pd_value_check;

-- Drop any CHECK constraint on lgd_value
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_lgd_value_check;

-- Drop any CHECK constraint on ead_value
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_ead_value_check;

-- Drop any CHECK constraint on ecl_value
ALTER TABLE ecl_segment_calculation 
DROP CONSTRAINT IF EXISTS ecl_segment_calculation_ecl_value_check;

-- ============================================================
-- PART 2: VERIFY CHANGES
-- ============================================================

-- Check remaining constraints
DO $$
DECLARE
    constraint_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO constraint_count
    FROM information_schema.table_constraints
    WHERE table_name = 'ecl_segment_calculation'
    AND constraint_type = 'CHECK';
    
    IF constraint_count = 0 THEN
        RAISE NOTICE '[SUCCESS] No CHECK constraints remain on ecl_segment_calculation';
    ELSE
        RAISE NOTICE '[WARNING] % CHECK constraints still exist on ecl_segment_calculation', constraint_count;
    END IF;
END $$;

-- ============================================================
-- ROLLBACK SCRIPT (for reference, do not execute)
-- ============================================================

/*
-- To rollback, you would need to know the original constraint definitions
-- Example (hypothetical):
ALTER TABLE ecl_segment_calculation 
ADD CONSTRAINT ecl_segment_calculation_segment_name_check 
CHECK (segment_name IN ('loan_intent', 'person_gender', 'person_education'));
*/

