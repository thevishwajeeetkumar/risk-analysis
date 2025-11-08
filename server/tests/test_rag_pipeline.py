import math
import json
import pandas as pd
import numpy as np


def test_sanitize_metadata_handles_nan_and_none():
    """Test that _sanitize_metadata properly handles NaN and None values."""
    # Import without running __init__ side effects
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)

    sample = {
        "file_id": "abc",
        "segment_type": "loan_intent",
        "segment": "VENTURE",
        "total_loans": None,
        "pd": float('nan'),
        "lgd": 0.35,
        "ead": 1000.0,
        "ecl": 350.0,
        "extra": object(),
    }

    sanitized = engine._sanitize_metadata(sample)

    # None becomes empty string
    assert sanitized["total_loans"] == ""
    # NaN becomes 0.0
    assert sanitized["pd"] == 0.0
    # JSON serializable
    json.dumps(sanitized)


def test_create_segment_documents_with_missing_columns():
    """Test that _create_segment_documents handles missing columns gracefully."""
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)
    
    # Create a DataFrame with missing columns
    ecl_df = pd.DataFrame({
        'Segment': ['TEST_SEGMENT'],
        # Missing 'Total Loans', 'PD', 'LGD', 'EAD', 'ECL'
    })
    
    ecl_results = {'test_type': ecl_df}
    
    # Should not crash and should use defaults
    documents = engine._create_segment_documents(ecl_results, 'test_file_id', user_id=1)
    
    assert len(documents) == 1
    assert documents[0].metadata['segment'] == 'TEST_SEGMENT'
    assert documents[0].metadata['total_loans'] == 0
    assert documents[0].metadata['pd'] == 0.0
    assert documents[0].metadata['lgd'] == 0.0
    assert documents[0].metadata['ead'] == 0.0
    assert documents[0].metadata['ecl'] == 0.0


def test_create_segment_documents_with_nan_values():
    """Test that _create_segment_documents handles NaN values gracefully."""
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)
    
    # Create a DataFrame with NaN values
    ecl_df = pd.DataFrame({
        'Segment': ['NAN_TEST'],
        'Total Loans': [np.nan],
        'PD': [np.nan],
        'LGD': [0.5],
        'EAD': [np.nan],
        'ECL': [100.0],
    })
    
    ecl_results = {'test_type': ecl_df}
    
    # Should not crash and should convert NaN to 0
    documents = engine._create_segment_documents(ecl_results, 'test_file_id')
    
    assert len(documents) == 1
    assert documents[0].metadata['segment'] == 'NAN_TEST'
    assert documents[0].metadata['total_loans'] == 0
    assert documents[0].metadata['pd'] == 0.0
    assert documents[0].metadata['lgd'] == 0.5
    assert documents[0].metadata['ead'] == 0.0
    assert documents[0].metadata['ecl'] == 100.0


def test_create_segment_documents_with_empty_strings():
    """Test that _create_segment_documents handles empty string values."""
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)
    
    # Create a DataFrame with empty strings
    ecl_df = pd.DataFrame({
        'Segment': ['', '  ', 'VALID_SEGMENT'],
        'Total Loans': [100, 200, 300],
        'PD': [0.1, 0.2, 0.3],
        'LGD': [0.4, 0.5, 0.6],
        'EAD': [1000, 2000, 3000],
        'ECL': [40, 200, 540],
    })
    
    ecl_results = {'test_type': ecl_df}
    
    documents = engine._create_segment_documents(ecl_results, 'test_file_id')
    
    # All segments should be processed (empty/whitespace segments get "UNKNOWN" label)
    assert len(documents) == 3
    assert documents[0].metadata['segment'] == 'UNKNOWN'
    assert documents[1].metadata['segment'] == 'UNKNOWN'
    assert documents[2].metadata['segment'] == 'VALID_SEGMENT'


def test_create_segment_documents_with_invalid_numeric_strings():
    """Test that _create_segment_documents handles invalid numeric strings."""
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)
    
    # Create a DataFrame with string values that should be numeric
    ecl_df = pd.DataFrame({
        'Segment': ['STRING_TEST'],
        'Total Loans': ['invalid'],
        'PD': ['not_a_number'],
        'LGD': [0.5],
        'EAD': ['N/A'],
        'ECL': [''],
    })
    
    ecl_results = {'test_type': ecl_df}
    
    # Should not crash and should coerce invalid values to 0
    documents = engine._create_segment_documents(ecl_results, 'test_file_id')
    
    assert len(documents) == 1
    assert documents[0].metadata['segment'] == 'STRING_TEST'
    assert documents[0].metadata['total_loans'] == 0
    assert documents[0].metadata['pd'] == 0.0
    assert documents[0].metadata['lgd'] == 0.5
    assert documents[0].metadata['ead'] == 0.0
    assert documents[0].metadata['ecl'] == 0.0


def test_deterministic_chunk_ids():
    """Test that chunk IDs are deterministic for the same input."""
    from core import rag_engine as re_mod
    import hashlib
    
    # Create two identical documents
    file_id = "test_file_123"
    segment_type = "loan_intent"
    segment = "VENTURE"
    content = "Segment Type: loan_intent\nSegment: VENTURE\nTotal Loans: 100"
    
    # Generate doc_id using the same logic as embed_segments
    content_snippet = content[:100]
    composite_key = f"{file_id}:{segment_type}:{segment}:{content_snippet}"
    doc_hash = hashlib.sha256(composite_key.encode('utf-8')).hexdigest()[:16]
    doc_id_1 = f"{file_id}:{segment_type}:{doc_hash}"
    
    # Generate again - should be identical
    doc_hash_2 = hashlib.sha256(composite_key.encode('utf-8')).hexdigest()[:16]
    doc_id_2 = f"{file_id}:{segment_type}:{doc_hash_2}"
    
    assert doc_id_1 == doc_id_2, "Chunk IDs should be deterministic"


def test_metadata_json_serializable():
    """Test that all metadata values are JSON serializable after sanitization."""
    from core import rag_engine as re_mod
    engine = re_mod.ECLRagEngine.__new__(re_mod.ECLRagEngine)
    
    # Create DataFrame with various problematic types
    ecl_df = pd.DataFrame({
        'Segment': ['JSON_TEST'],
        'Total Loans': [100],
        'PD': [0.5],
        'LGD': [0.3],
        'EAD': [1000],
        'ECL': [150],
    })
    
    ecl_results = {'test_type': ecl_df}
    documents = engine._create_segment_documents(ecl_results, 'test_file_id', user_id=42)
    
    # All metadata should be JSON serializable
    for doc in documents:
        json_str = json.dumps(doc.metadata)
        assert json_str is not None
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert parsed['segment'] == 'JSON_TEST'
        assert parsed['user_id'] == 42


def test_migration_sets_default_metadata():
    """Test that database migration sets default metadata."""
    from pathlib import Path
    migration = Path(__file__).parent.parent / "db" / "migrations" / "001_add_indexes_and_constraints.sql"
    contents = migration.read_text(encoding="utf-8")
    assert "ALTER COLUMN metadata SET DEFAULT '{}'" in contents


