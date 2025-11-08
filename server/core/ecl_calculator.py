"""
ECL (Expected Credit Loss) Calculator Module

Implements direct ECL calculation using the formula:
    PD (Probability of Default) = proportion of defaults in segment
    LGD (Loss Given Default) = 0.35 + (avg_interest_rate / 100) * 0.05
    EAD (Exposure at Default) = average loan amount in segment
    ECL = PD × LGD × EAD

This is a statistical approach based on historical default rates,
not machine learning prediction.
"""

import pandas as pd
import numpy as np
from typing import Union, List
from core.config import BASE_LGD, LGD_RATE_MULTIPLIER, MIN_SEGMENT_SIZE


def calculate_ecl(df: pd.DataFrame, segment_col: Union[str, List[str]]) -> pd.DataFrame:
    """
    Calculate PD, LGD, EAD, and ECL for each segment.
    
    Args:
        df: DataFrame with loan data including loan_status, loan_amnt, loan_int_rate
        segment_col: Column name(s) to segment by (str or list of str)
    
    Returns:
        DataFrame with columns: Segment, Total Loans, PD, LGD, EAD, ECL
        Sorted by ECL in descending order (highest risk first)
    
    Formula:
        PD = (loan_status == 0).mean()  # proportion of defaults
        LGD = 0.35 + (avg_interest_rate / 100) * 0.05
        EAD = loan_amnt.mean()
        ECL = PD × LGD × EAD
    """
    
    # Validate input
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")
    
    # Handle single column or multiple columns
    if isinstance(segment_col, str):
        segment_cols = [segment_col]
    elif isinstance(segment_col, list) and all(c in df.columns for c in segment_col):
        segment_cols = segment_col
    else:
        raise ValueError(f"Invalid segment column(s): {segment_col}")
    
    # Check required columns
    required_cols = ["loan_status", "loan_amnt"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert numeric columns safely
    numeric_cols = ["loan_amnt", "loan_int_rate", "credit_score", "person_income"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Fill missing values for calculation with fallback if mean is NaN
    if "loan_int_rate" in df.columns:
        mean_rate = df["loan_int_rate"].mean()
        if pd.isna(mean_rate):
            mean_rate = 10.0  # Default: 10% interest rate
            print(f"[WARNING]  Warning: All loan_int_rate values are NULL, using default: {mean_rate}%")
        df = df.fillna({"loan_int_rate": mean_rate})
    
    mean_amt = df["loan_amnt"].mean()
    if pd.isna(mean_amt):
        mean_amt = 10000.0  # Default: $10,000 loan amount
        print(f"[WARNING]  Warning: All loan_amnt values are NULL, using default: ${mean_amt:,.0f}")
    df = df.fillna({"loan_amnt": mean_amt})
    
    # Validate segment columns exist
    for seg in segment_cols:
        if seg not in df.columns:
            raise ValueError(f"Invalid segment column: {seg}")

    # Group by segment column(s)
    results = []
    grouped = df.groupby(segment_cols)
    
    for segment, group in grouped:
        total_loans = len(group)
        
        # Skip empty groups
        if total_loans == 0:
            print(f"[WARNING] Skipping segment '{segment}' - no loans in group")
            continue
        
        # Skip segments with insufficient data for statistical significance
        if total_loans < MIN_SEGMENT_SIZE:
            print(f"[WARNING] Segment '{segment}' has only {total_loans} loans (< {MIN_SEGMENT_SIZE}); skipping")
            continue
        
        # Handle tuple for multiple grouping columns
        if isinstance(segment, tuple):
            segment_name = ", ".join(map(str, segment))
        else:
            segment_name = str(segment)
        
        # PD = Probability of Default (proportion where loan_status == 0)
        pd_value = (group["loan_status"] == 0).mean()
        
        # LGD = Loss Given Default
        # Base LGD + interest rate adjustment
        if "loan_int_rate" in group.columns:
            avg_rate = group["loan_int_rate"].mean()
            lgd_value = BASE_LGD + (avg_rate / 100) * LGD_RATE_MULTIPLIER
        else:
            lgd_value = BASE_LGD
        
        # EAD = Exposure at Default (average loan amount)
        ead_value = group["loan_amnt"].mean()
        
        # ECL = Expected Credit Loss
        ecl = pd_value * lgd_value * ead_value
        
        results.append({
            "Segment": segment_name,
            "Total Loans": total_loans,
            "PD": round(pd_value, 4),
            "LGD": round(lgd_value, 4),
            "EAD": round(ead_value, 2),
            "ECL": round(ecl, 2)
        })
    
    if not results:
        raise ValueError("No valid segments found in dataset.")
    
    # Create DataFrame and sort by ECL (highest risk first)
    result_df = pd.DataFrame(results).sort_values(by="ECL", ascending=False)
    return result_df


def calculate_all_segments(df: pd.DataFrame, segment_columns: List[str]) -> dict:
    """
    Calculate ECL for all specified segment types.
    
    Args:
        df: DataFrame with loan data
        segment_columns: List of column names to segment by
    
    Returns:
        Dictionary mapping segment_type to ECL DataFrame
        Example: {"loan_intent": df1, "person_gender": df2, ...}
    """
    results = {}
    
    for col in segment_columns:
        if col in df.columns:
            try:
                ecl_df = calculate_ecl(df, col)
                results[col] = ecl_df
            except Exception as e:
                print(f"Warning: Could not calculate ECL for segment '{col}': {e}")
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    return results


def get_segment_summary(ecl_dict: dict) -> pd.DataFrame:
    """
    Create a summary of all segments across all types.
    
    Args:
        ecl_dict: Dictionary from calculate_all_segments()
    
    Returns:
        DataFrame with all segments combined, including segment_type column
    """
    all_segments = []
    
    for segment_type, ecl_df in ecl_dict.items():
        df_copy = ecl_df.copy()
        df_copy["Segment Type"] = segment_type
        all_segments.append(df_copy)
    
    if not all_segments:
        return pd.DataFrame()
    
    combined = pd.concat(all_segments, ignore_index=True)
    
    # Reorder columns to put Segment Type first
    cols = ["Segment Type", "Segment", "Total Loans", "PD", "LGD", "EAD", "ECL"]
    combined = combined[cols]
    
    # Sort by ECL descending
    combined = combined.sort_values(by="ECL", ascending=False)
    
    return combined

