"""
Data Segmentation Module

Segments cleaned loan data by various categorical columns
for ECL calculation and analysis.
"""

import pandas as pd
from typing import List, Dict
from pathlib import Path

from core.config import SEGMENT_COLUMNS, SEGMENTS_DIR
from core.ecl_calculator import calculate_ecl, calculate_all_segments, get_segment_summary


def segment_and_calculate_ecl(
    df: pd.DataFrame,
    segment_columns: List[str] = None,
    save_to_csv: bool = True,
    file_id: str = "default"
) -> Dict[str, pd.DataFrame]:
    """
    Segment data by specified columns and calculate ECL for each segment.
    
    Args:
        df: Cleaned DataFrame with loan data
        segment_columns: List of columns to segment by. If None, uses config.SEGMENT_COLUMNS
        save_to_csv: Whether to save segment results as CSV files
        file_id: Identifier for the uploaded file (used in filenames)
    
    Returns:
        Dictionary mapping segment_type to ECL DataFrame
        Example: {
            "loan_intent": DataFrame with ECL by loan intent,
            "person_gender": DataFrame with ECL by gender,
            ...
        }
    """
    if segment_columns is None:
        segment_columns = SEGMENT_COLUMNS
    
    # Calculate ECL for all segments
    ecl_results = calculate_all_segments(df, segment_columns)
    
    # Save to CSV files if requested
    if save_to_csv:
        save_segment_results(ecl_results, file_id)
    
    return ecl_results


def save_segment_results(ecl_results: Dict[str, pd.DataFrame], file_id: str = "default"):
    """
    Save ECL segment results to CSV files.
    
    Args:
        ecl_results: Dictionary from segment_and_calculate_ecl()
        file_id: Identifier for the uploaded file
    """
    # Ensure segments directory exists
    Path(SEGMENTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save individual segment files
    for segment_type, ecl_df in ecl_results.items():
        filename = f"ecl_by_{segment_type}_{file_id}.csv"
        filepath = Path(SEGMENTS_DIR) / filename
        ecl_df.to_csv(filepath, index=False)
        print(f"[SUCCESS] Saved segment data to {filepath}")
    
    # Save combined summary
    summary_df = get_segment_summary(ecl_results)
    summary_filename = f"ecl_all_segments_{file_id}.csv"
    summary_filepath = Path(SEGMENTS_DIR) / summary_filename
    summary_df.to_csv(summary_filepath, index=False)
    print(f"[SUCCESS] Saved segment summary to {summary_filepath}")


def load_segment_results(file_id: str = "default") -> Dict[str, pd.DataFrame]:
    """
    Load saved segment results from CSV files.
    
    Args:
        file_id: Identifier for the uploaded file
    
    Returns:
        Dictionary mapping segment_type to ECL DataFrame
    """
    results = {}
    
    for segment_type in SEGMENT_COLUMNS:
        filename = f"ecl_by_{segment_type}_{file_id}.csv"
        filepath = Path(SEGMENTS_DIR) / filename
        
        if filepath.exists():
            results[segment_type] = pd.read_csv(filepath)
        else:
            print(f"Warning: Segment file not found: {filepath}")
    
    return results


def get_segment_statistics(ecl_results: Dict[str, pd.DataFrame]) -> dict:
    """
    Calculate summary statistics across all segments.
    
    Args:
        ecl_results: Dictionary from segment_and_calculate_ecl()
    
    Returns:
        Dictionary with summary statistics
    """
    summary = get_segment_summary(ecl_results)
    
    if summary.empty:
        return {}
    
    stats = {
        "total_segments": len(summary),
        "total_loans": int(summary["Total Loans"].sum()),
        "avg_pd": float(summary["PD"].mean()),
        "avg_lgd": float(summary["LGD"].mean()),
        "avg_ead": float(summary["EAD"].mean()),
        "total_ecl": float(summary["ECL"].sum()),
        "avg_ecl": float(summary["ECL"].mean()),
        "max_ecl_segment": summary.iloc[0]["Segment"] if len(summary) > 0 else None,
        "max_ecl_value": float(summary.iloc[0]["ECL"]) if len(summary) > 0 else 0,
        "segment_types": list(ecl_results.keys())
    }
    
    return stats

