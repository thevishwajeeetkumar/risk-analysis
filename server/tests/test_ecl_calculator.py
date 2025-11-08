"""
Tests for ECL Calculator

Validates that ECL calculations match the proposed formula:
    PD = (loan_status == 0).mean()
    LGD = 0.35 + (avg_rate / 100) * 0.05
    EAD = loan_amnt.mean()
    ECL = PD × LGD × EAD
"""

import pytest
import pandas as pd
import numpy as np
from core.ecl_calculator import calculate_ecl, calculate_all_segments, get_segment_summary


def test_ecl_formula_basic():
    """Test ECL calculation with known values."""
    # Create sample data
    data = {
        'loan_status': [0, 0, 1, 0, 1],  # 3 defaults out of 5 = 60% PD
        'loan_amnt': [10000, 10000, 10000, 10000, 10000],  # Average = 10000
        'loan_int_rate': [10, 10, 10, 10, 10],  # Average = 10%
        'loan_intent': ['PERSONAL', 'PERSONAL', 'PERSONAL', 'PERSONAL', 'PERSONAL']
    }
    df = pd.DataFrame(data)
    
    # Calculate ECL
    result = calculate_ecl(df, 'loan_intent')
    
    # Expected values
    expected_pd = 0.6  # 3/5 = 0.6
    expected_lgd = 0.35 + (10 / 100) * 0.05  # 0.35 + 0.005 = 0.355
    expected_ead = 10000
    expected_ecl = expected_pd * expected_lgd * expected_ead  # 0.6 * 0.355 * 10000 = 2130
    
    # Validate
    assert len(result) == 1
    assert result.iloc[0]['Segment'] == 'PERSONAL'
    assert result.iloc[0]['Total Loans'] == 5
    assert abs(result.iloc[0]['PD'] - expected_pd) < 0.0001
    assert abs(result.iloc[0]['LGD'] - expected_lgd) < 0.0001
    assert abs(result.iloc[0]['EAD'] - expected_ead) < 0.01
    assert abs(result.iloc[0]['ECL'] - expected_ecl) < 0.01


def test_ecl_output_format():
    """Test that output format matches proposed solution."""
    data = {
        'loan_status': [0, 1, 0, 1, 0, 0],
        'loan_amnt': [5000, 10000, 7500, 12000, 8000, 9000],
        'loan_int_rate': [12, 15, 10, 18, 14, 11],
        'loan_intent': ['EDUCATION', 'PERSONAL', 'EDUCATION', 'PERSONAL', 'EDUCATION', 'EDUCATION']
    }
    df = pd.DataFrame(data)
    
    result = calculate_ecl(df, 'loan_intent')
    
    # Check columns match proposed format
    expected_columns = ['Segment', 'Total Loans', 'PD', 'LGD', 'EAD', 'ECL']
    assert list(result.columns) == expected_columns
    
    # Check sorting (highest ECL first)
    assert result['ECL'].is_monotonic_decreasing


def test_high_default_segment():
    """Test segment with very high default rate (like VENTURE in sample)."""
    data = {
        'loan_status': [0] * 85 + [1] * 15,  # 85% default rate
        'loan_amnt': [10000] * 100,
        'loan_int_rate': [16] * 100,
        'loan_intent': ['VENTURE'] * 100
    }
    df = pd.DataFrame(data)
    
    result = calculate_ecl(df, 'loan_intent')
    
    assert result.iloc[0]['PD'] == 0.85
    assert result.iloc[0]['Total Loans'] == 100
    # LGD = 0.35 + (16/100) * 0.05 = 0.35 + 0.008 = 0.358
    assert abs(result.iloc[0]['LGD'] - 0.358) < 0.001
    

def test_multiple_segments():
    """Test calculation across multiple segments."""
    data = {
        'loan_status': [0, 1, 0, 1, 0, 1, 0, 1],
        'loan_amnt': [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
        'loan_int_rate': [10, 12, 14, 16, 10, 12, 14, 16],
        'loan_intent': ['EDUCATION', 'EDUCATION', 'PERSONAL', 'PERSONAL', 
                       'MEDICAL', 'MEDICAL', 'VENTURE', 'VENTURE']
    }
    df = pd.DataFrame(data)
    
    result = calculate_ecl(df, 'loan_intent')
    
    # Should have 4 segments
    assert len(result) == 4
    
    # Check all segments present
    segments = set(result['Segment'])
    assert segments == {'EDUCATION', 'PERSONAL', 'MEDICAL', 'VENTURE'}
    
    # Each segment should have 2 loans
    assert all(result['Total Loans'] == 2)


def test_calculate_all_segments():
    """Test calculating ECL for all segment types."""
    data = {
        'loan_status': [0, 1, 0, 1] * 3,
        'loan_amnt': [10000] * 12,
        'loan_int_rate': [12] * 12,
        'loan_intent': ['PERSONAL'] * 6 + ['EDUCATION'] * 6,
        'person_gender': ['Male'] * 3 + ['Female'] * 3 + ['Male'] * 3 + ['Female'] * 3
    }
    df = pd.DataFrame(data)
    
    segment_cols = ['loan_intent', 'person_gender']
    results = calculate_all_segments(df, segment_cols)
    
    # Should have 2 result DataFrames
    assert len(results) == 2
    assert 'loan_intent' in results
    assert 'person_gender' in results
    
    # loan_intent should have 2 segments
    assert len(results['loan_intent']) == 2
    
    # person_gender should have 2 segments
    assert len(results['person_gender']) == 2


def test_get_segment_summary():
    """Test combined segment summary."""
    data = {
        'loan_status': [0, 1, 0, 1] * 2,
        'loan_amnt': [10000] * 8,
        'loan_int_rate': [12] * 8,
        'loan_intent': ['PERSONAL'] * 4 + ['EDUCATION'] * 4,
        'person_gender': ['Male', 'Female'] * 4
    }
    df = pd.DataFrame(data)
    
    ecl_results = calculate_all_segments(df, ['loan_intent', 'person_gender'])
    summary = get_segment_summary(ecl_results)
    
    # Should combine all segments
    assert len(summary) == 4  # 2 intent + 2 gender
    
    # Should have Segment Type column
    assert 'Segment Type' in summary.columns
    
    # Should be sorted by ECL
    assert summary['ECL'].is_monotonic_decreasing


def test_edge_case_all_defaults():
    """Test when all loans default."""
    data = {
        'loan_status': [0] * 5,
        'loan_amnt': [10000] * 5,
        'loan_int_rate': [15] * 5,
        'loan_intent': ['VENTURE'] * 5
    }
    df = pd.DataFrame(data)
    
    result = calculate_ecl(df, 'loan_intent')
    
    assert result.iloc[0]['PD'] == 1.0  # 100% default


def test_edge_case_no_defaults():
    """Test when no loans default."""
    data = {
        'loan_status': [1] * 5,
        'loan_amnt': [10000] * 5,
        'loan_int_rate': [10] * 5,
        'loan_intent': ['EDUCATION'] * 5
    }
    df = pd.DataFrame(data)
    
    result = calculate_ecl(df, 'loan_intent')
    
    assert result.iloc[0]['PD'] == 0.0  # 0% default
    assert result.iloc[0]['ECL'] == 0.0  # No expected loss


def test_missing_required_column():
    """Test error handling for missing required columns."""
    data = {
        'loan_amnt': [10000] * 5,
        'loan_intent': ['PERSONAL'] * 5
        # Missing loan_status
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_ecl(df, 'loan_intent')


def test_empty_dataframe():
    """Test error handling for empty DataFrame."""
    df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="empty"):
        calculate_ecl(df, 'loan_intent')


def test_invalid_segment_column():
    """Test error handling for invalid segment column."""
    data = {
        'loan_status': [0, 1],
        'loan_amnt': [10000, 10000],
        'loan_intent': ['PERSONAL', 'PERSONAL']
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Invalid segment column"):
        calculate_ecl(df, 'nonexistent_column')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

