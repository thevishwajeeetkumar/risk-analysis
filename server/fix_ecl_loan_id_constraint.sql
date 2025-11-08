-- Fix: Allow NULL values in ecl_segment_calculation.loan_id
-- Reason: Aggregate ECL calculations don't link to specific loans

-- Remove NOT NULL constraint from loan_id column
ALTER TABLE ecl_segment_calculation 
ALTER COLUMN loan_id DROP NOT NULL;

-- Verify the change
SELECT column_name, is_nullable, data_type 
FROM information_schema.columns 
WHERE table_name = 'ecl_segment_calculation' 
AND column_name = 'loan_id';

-- Expected result: is_nullable = 'YES'

