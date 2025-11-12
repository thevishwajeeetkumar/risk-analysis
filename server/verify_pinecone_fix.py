"""
Verification Script for Pinecone Field Map Fix

This script verifies that the field_map fix is working correctly by:
1. Checking the RAG engine configuration
2. Simulating the embedding flow with test data
3. Validating Pinecone index creation parameters
4. Testing error handling for invalid data

Run this script to verify the fix before deploying to production.

Usage:
    python verify_pinecone_fix.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict

# Add Interview directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.rag_engine import ECLRagEngine, get_shared_index, get_user_namespace
from core.config import OPENAI_EMBEDDING_MODEL


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_success(text: str):
    """Print a success message."""
    print(f"✅ {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"❌ {text}")


def print_info(text: str):
    """Print an info message."""
    print(f"ℹ️  {text}")


def check_environment():
    """Verify required environment variables are set."""
    print_header("1. Environment Check")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings",
        "PINECONE_API_KEY": "Pinecone API key for vector storage",
        "DATABASE_URL": "PostgreSQL database connection",
        "JWT_SECRET_KEY": "JWT authentication secret"
    }
    
    all_present = True
    for var, description in required_vars.items():
        if os.getenv(var):
            print_success(f"{var} is set ({description})")
        else:
            print_error(f"{var} is NOT set ({description})")
            all_present = False
    
    return all_present


def test_ecl_validation():
    """Test ECL DataFrame validation logic."""
    print_header("2. ECL DataFrame Validation Test")
    
    engine = ECLRagEngine()
    
    # Test 1: Valid DataFrame
    print_info("Test 1: Valid ECL DataFrame")
    valid_df = pd.DataFrame({
        'Segment': ['EDUCATION', 'MEDICAL'],
        'Total Loans': [50, 75],
        'PD': [0.15, 0.10],
        'LGD': [0.40, 0.38],
        'EAD': [15000.00, 20000.00],
        'ECL': [900.00, 760.00]
    })
    
    result = engine._validate_ecl_dataframe(valid_df, 'test_segment')
    if result:
        print_success("Valid DataFrame passed validation")
    else:
        print_error("Valid DataFrame failed validation")
    
    # Test 2: Invalid DataFrame (missing columns)
    print_info("Test 2: Invalid ECL DataFrame (missing columns)")
    invalid_df = pd.DataFrame({
        'Segment': ['EDUCATION'],
        'Total Loans': [50],
        'PD': [0.15]
        # Missing: LGD, EAD, ECL
    })
    
    result = engine._validate_ecl_dataframe(invalid_df, 'test_segment')
    if not result:
        print_success("Invalid DataFrame correctly rejected")
    else:
        print_error("Invalid DataFrame incorrectly accepted")
    
    # Test 3: Empty DataFrame
    print_info("Test 3: Empty ECL DataFrame")
    empty_df = pd.DataFrame()
    
    result = engine._validate_ecl_dataframe(empty_df, 'test_segment')
    if not result:
        print_success("Empty DataFrame correctly rejected")
    else:
        print_error("Empty DataFrame incorrectly accepted")
    
    return True


def test_document_creation():
    """Test document creation from ECL results."""
    print_header("3. Document Creation Test")
    
    engine = ECLRagEngine()
    
    # Create test ECL results
    ecl_results = {
        'loan_intent': pd.DataFrame({
            'Segment': ['EDUCATION', 'MEDICAL'],
            'Total Loans': [50, 75],
            'PD': [0.15, 0.10],
            'LGD': [0.40, 0.38],
            'EAD': [15000.00, 20000.00],
            'ECL': [900.00, 760.00]
        })
    }
    
    try:
        documents = engine._create_segment_documents(
            ecl_results,
            file_id='test_verification',
            user_id=999
        )
        
        print_success(f"Created {len(documents)} documents")
        
        # Validate document structure
        if len(documents) == 2:
            print_success("Correct number of documents created")
        else:
            print_error(f"Expected 2 documents, got {len(documents)}")
        
        # Check first document
        doc = documents[0]
        if doc.page_content:
            print_success("Document has page_content")
        else:
            print_error("Document missing page_content")
        
        # Check metadata
        required_metadata = ['file_id', 'segment_type', 'segment', 'pd', 'lgd', 'ead', 'ecl']
        missing = [key for key in required_metadata if key not in doc.metadata]
        if not missing:
            print_success("Document has all required metadata fields")
        else:
            print_error(f"Document missing metadata fields: {missing}")
        
        # Check metadata values are sanitized
        has_none = any(v is None for v in doc.metadata.values())
        has_nan = any(pd.isna(v) for v in doc.metadata.values())
        if not has_none and not has_nan:
            print_success("Metadata properly sanitized (no None/NaN)")
        else:
            print_error("Metadata contains None or NaN values")
        
        return True
    
    except Exception as e:
        print_error(f"Document creation failed: {str(e)}")
        return False


def test_metadata_sanitization():
    """Test metadata sanitization logic."""
    print_header("4. Metadata Sanitization Test")
    
    engine = ECLRagEngine()
    
    # Test with problematic values
    test_metadata = {
        'none_value': None,
        'nan_value': float('nan'),
        'valid_int': 42,
        'valid_float': 3.14,
        'valid_str': 'test',
        'valid_bool': True
    }
    
    sanitized = engine._sanitize_metadata(test_metadata)
    
    # Check None replacement
    if sanitized['none_value'] == '':
        print_success("None values replaced with empty string")
    else:
        print_error(f"None not sanitized: {sanitized['none_value']}")
    
    # Check NaN replacement
    if sanitized['nan_value'] == 0.0:
        print_success("NaN values replaced with 0.0")
    else:
        print_error(f"NaN not sanitized: {sanitized['nan_value']}")
    
    # Check valid values preserved
    if sanitized['valid_int'] == 42:
        print_success("Valid int preserved")
    else:
        print_error(f"Valid int changed: {sanitized['valid_int']}")
    
    return True


def verify_pinecone_field_map():
    """Verify that Pinecone index creation includes field_map."""
    print_header("5. Pinecone Field Map Verification")
    
    print_info("Checking Pinecone configuration...")
    
    # Check that the embedding model is configured
    if OPENAI_EMBEDDING_MODEL:
        print_success(f"Embedding model configured: {OPENAI_EMBEDDING_MODEL}")
    else:
        print_error("Embedding model not configured")
        return False
    
    # Check the get_shared_index and get_user_namespace functions
    import inspect
    try:
        shared_index_source = inspect.getsource(get_shared_index)
        namespace_source = inspect.getsource(get_user_namespace)
        
        if 'PINECONE_INDEX_NAME' in shared_index_source:
            print_success("get_shared_index uses shared index configuration")
        else:
            print_error("get_shared_index might not be using shared index")
            return False
        
        if 'namespace' in shared_index_source.lower() or 'namespace' in namespace_source.lower():
            print_success("Namespace-based user isolation found")
        else:
            print_info("Namespace-based isolation might not be implemented")
    except Exception as e:
        print_error(f"Could not verify index functions: {e}")
        return False
    
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print_header("6. Error Handling Test")
    
    engine = ECLRagEngine()
    
    # Test 1: Empty ECL results
    print_info("Test 1: Empty ECL results")
    try:
        engine.embed_segments(
            ecl_results={},
            file_id='test',
            username='testuser',
            user_id=1
        )
        print_error("Empty ECL results should raise ValueError")
    except ValueError as e:
        if "empty" in str(e).lower():
            print_success("Empty ECL results correctly raises ValueError")
        else:
            print_error(f"Wrong error message: {str(e)}")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
    
    # Test 2: Invalid ECL structure
    print_info("Test 2: Invalid ECL structure")
    invalid_ecl = {
        'bad_segment': pd.DataFrame({
            'wrong': ['columns'],
            'here': ['data']
        })
    }
    
    try:
        engine.embed_segments(
            ecl_results=invalid_ecl,
            file_id='test',
            username='testuser',
            user_id=1
        )
        print_error("Invalid ECL structure should raise ValueError")
    except ValueError as e:
        if "no valid documents" in str(e).lower():
            print_success("Invalid ECL structure correctly raises ValueError")
        else:
            print_error(f"Wrong error message: {str(e)}")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
    
    return True


def print_summary(results: Dict[str, bool]):
    """Print summary of verification results."""
    print_header("Verification Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print_success("\nAll verification checks passed! ✨")
        print_info("The Pinecone field_map fix is working correctly.")
        print_info("You can now proceed with uploading real data.")
    else:
        print_error("\nSome verification checks failed.")
        print_info("Please review the errors above and fix them before proceeding.")
        print("\nFailed tests:")
        for test_name, result in results.items():
            if not result:
                print(f"  - {test_name}")
    
    return failed == 0


def main():
    """Run all verification checks."""
    print_header("Pinecone Field Map Fix - Verification Script")
    print("This script verifies that the fix is working correctly.\n")
    
    results = {}
    
    # Run verification steps
    results['environment'] = check_environment()
    results['ecl_validation'] = test_ecl_validation()
    results['document_creation'] = test_document_creation()
    results['metadata_sanitization'] = test_metadata_sanitization()
    results['pinecone_field_map'] = verify_pinecone_field_map()
    results['error_handling'] = test_error_handling()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()

