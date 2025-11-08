"""
Data Preprocessing Module

Steps:
1. Load CSV/XLSX files
2. Handle missing values (median for numeric, mode for categorical)
3. Cap outliers using IQR method
4. Return cleaned DataFrame

This is the first step in the ECL calculation pipeline.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from core.config import IQR_COLS, PROCESSED_DIR, AGE_GROUPS


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV or XLSX file.
    
    Args:
        path: Path to the data file (CSV or XLSX)
    
    Returns:
        DataFrame with stripped column names and values
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load based on file extension
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or XLSX.")
    
    # Strip whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    
    print(f"[SUCCESS] Data loaded: {df.shape}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numeric columns: median
    - Categorical columns: mode (or "UNKNOWN" if no mode)
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Numeric columns: fill with median
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {col} (numeric) with median: {median_val:.2f}")
    
    # Categorical columns: fill with mode or "UNKNOWN"
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_value = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
            df[col].fillna(fill_value, inplace=True)
            print(f"  Imputed {col} (categorical) with: {fill_value}")
    
    return df


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age_group column from person_age.
    
    Converts continuous age values to categorical segments:
    - young: 0-35 years
    - middle_aged: 35-55 years  
    - senior_citizen: 55+ years
    
    Args:
        df: DataFrame containing person_age column
    
    Returns:
        DataFrame with added age_group column
    """
    df = df.copy()
    
    if 'person_age' not in df.columns:
        print("  Warning: person_age column not found, skipping age grouping")
        return df
    
    def categorize_age(age):
        """Categorize age into groups."""
        for group_name, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age < max_age:
                return group_name
        return "unknown"
    
    df['age_group'] = df['person_age'].apply(categorize_age)
    
    # Print distribution
    age_dist = df['age_group'].value_counts()
    print("  Age group distribution:")
    for group, count in age_dist.items():
        print(f"    {group}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def cap_outliers_iqr(series: pd.Series, whisker: float = 1.5) -> pd.Series:
    """
    Cap outliers using IQR (Interquartile Range) method.
    
    Formula:
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower bound = Q1 - whisker * IQR
        Upper bound = Q3 + whisker * IQR
    
    Args:
        series: Numeric series to cap
        whisker: IQR multiplier (default 1.5)
    
    Returns:
        Series with capped values
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - whisker * iqr, q3 + whisker * iqr
    
    n_outliers = ((series < lower) | (series > upper)).sum()
    if n_outliers > 0:
        print(f"  Capped {n_outliers} outliers in range [{lower:.2f}, {upper:.2f}]")
    
    return series.clip(lower, upper)


def cap_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Cap outliers for specified columns using IQR method.
    
    Args:
        df: Input DataFrame
        cols: List of columns to cap. If None, uses config.IQR_COLS
    
    Returns:
        DataFrame with capped outliers
    """
    if cols is None:
        cols = IQR_COLS
    
    df = df.copy()
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            print(f"Capping outliers for: {col}")
            df[col] = cap_outliers_iqr(df[col])
    
    return df


def preprocess_data(
    input_path: Union[str, Path],
    save_output: bool = True,
    output_filename: str = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline:
    1. Load data
    2. Impute missing values
    3. Cap outliers
    4. Optionally save to processed directory
    
    Args:
        input_path: Path to input file (CSV or XLSX)
        save_output: Whether to save cleaned data
        output_filename: Custom output filename (auto-generated if None)
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(input_path)
    
    # Step 2: Impute missing values
    print("\nImputing missing values...")
    df = impute_missing(df)
    
    # Step 3: Cap outliers
    print("\nCapping outliers...")
    df = cap_outliers(df)
    
    # Step 4: Create age groups
    print("\nCreating age groups...")
    df = create_age_groups(df)
    
    # Step 5: Save if requested
    if save_output:
        if output_filename is None:
            # Generate filename from input
            input_name = Path(input_path).stem
            output_filename = f"clean_{input_name}.csv"
        
        output_path = Path(PROCESSED_DIR) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Preprocessing complete -> {output_path}")
    else:
        print(f"\n[SUCCESS] Preprocessing complete (not saved)")
    
    return df


def main():
    """
    Standalone execution for testing/legacy compatibility.
    """
    # For backward compatibility
    DATA_PATH = "loan_data.csv"
    OUTPUT_PATH = "clean_loans.csv"
    
    if os.path.exists(DATA_PATH):
        df = preprocess_data(DATA_PATH, save_output=True, output_filename=OUTPUT_PATH)
        print(f"Final shape: {df.shape}")
    else:
        print(f"Error: {DATA_PATH} not found")


if __name__ == "__main__":
    main()

