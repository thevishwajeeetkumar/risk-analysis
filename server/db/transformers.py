"""
Data transformation utilities for converting between DataFrames and database models.

Provides functions to transform pandas DataFrames to SQLAlchemy models and vice versa.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from db.models import Loan, ECLSegmentCalculation
from core.preprocess import create_age_groups


# Column mapping between DataFrame and Database
DF_TO_DB_MAPPING = {
    "loan_amnt": "loan_amount",
    "person_home_ownership": "home_ownership",
    # Add any other mappings as needed
}
DB_TO_DF_MAPPING = {v: k for k, v in DF_TO_DB_MAPPING.items()}


def _to_bool(value: Any) -> bool:
    """Convert various truthy/falsy representations to boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1", "t"}:
            return True
        if normalized in {"no", "n", "false", "0", "f"}:
            return False
    # Fallback
    return False


def df_to_loan_records(df: pd.DataFrame, user_id: int) -> List[Dict[str, Any]]:
    """
    Convert DataFrame rows to Loan model dictionaries.
    
    Args:
        df: DataFrame with loan data (already cleaned)
        user_id: ID of the user who uploaded the data
        
    Returns:
        List of dictionaries ready for bulk insertion
    """
    loan_records = []
    failed_rows = []
    
    for idx, row in df.iterrows():
        try:
            loan_data = {
                "user_id": user_id,
                "loan_amount": float(row["loan_amnt"]),  # DF has loan_amnt, DB has loan_amount
                "loan_intent": str(row["loan_intent"]),
                "loan_int_rate": float(row["loan_int_rate"]),
                "loan_percent_income": float(row["loan_percent_income"]),
                "credit_score": int(row["credit_score"]),
                "person_income": float(row["person_income"]),
                "person_age": int(row["person_age"]),
                "person_gender": str(row["person_gender"]),
                "person_education": str(row["person_education"]),
                "person_emp_exp": int(row.get("person_emp_exp", 0)),  # May be missing
                "home_ownership": str(row["person_home_ownership"]),  # DF has person_home_ownership, DB has home_ownership
                "cb_person_cred_hist_length": int(row["cb_person_cred_hist_length"]),
                "previous_loan_defaults_on_file": _to_bool(row["previous_loan_defaults_on_file"]),
                "loan_status": int(row["loan_status"]),
                "created_at": datetime.utcnow()
            }
            
            loan_records.append(loan_data)
            
        except (ValueError, TypeError, KeyError) as e:
            failed_rows.append((idx, str(e)))
            print(f"[WARNING]  Skipping row {idx}: {str(e)}")
            continue
    
    if failed_rows:
        print(f"[WARNING]  Warning: {len(failed_rows)} of {len(df)} rows failed conversion")
        print(f"   Successfully converted: {len(loan_records)} rows")
        
        # Check failure threshold (fail if more than 50% rows are bad)
        failure_rate = len(failed_rows) / len(df)
        if failure_rate > 0.5:
            raise ValueError(f"Too many rows failed conversion: {len(failed_rows)}/{len(df)} ({failure_rate:.1%}). Please check your data format.")
    
    if not loan_records:
        raise ValueError("All rows failed conversion. Please check your data format.")
    
    return loan_records


def ecl_results_to_db_records(
    ecl_dict: Dict[str, pd.DataFrame],
    user_id: int,
    loan_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Convert ECL calculation results to ECLSegmentCalculation model dictionaries.
    
    Args:
        ecl_dict: Dictionary mapping segment types to ECL DataFrames
                 (output from segment_and_calculate_ecl)
        user_id: ID of the user who triggered the calculation
        loan_ids: Optional list of loan IDs if you want to link calculations
                 to specific loans (usually None for aggregate calculations)
        
    Returns:
        List of dictionaries ready for bulk insertion
    """
    ecl_records = []
    timestamp = datetime.utcnow()
    
    for segment_name, ecl_df in ecl_dict.items():
        for _, row in ecl_df.iterrows():
            ecl_data = {
                "user_id": user_id,
                "loan_id": None,  # Aggregate calculations don't link to specific loans
                "segment_name": segment_name,
                "segment_value": str(row["Segment"]),
                "pd_value": float(row["PD"]),
                "lgd_value": float(row["LGD"]),
                "ead_value": float(row["EAD"]),
                "ecl_value": float(row["ECL"]),
                "created_at": timestamp  # Same timestamp for all records in this batch
            }
            
            ecl_records.append(ecl_data)
    
    return ecl_records


def db_ecl_to_dict(ecl_records: List[ECLSegmentCalculation]) -> Dict[str, pd.DataFrame]:
    """
    Convert ECL database records back to dictionary format for API responses.
    
    Args:
        ecl_records: List of ECLSegmentCalculation model instances
        
    Returns:
        Dictionary mapping segment types to DataFrames
    """
    if not ecl_records:
        return {}
    
    # Group by segment_name
    segment_data = {}
    
    for record in ecl_records:
        segment_name = record.segment_name
        
        if segment_name not in segment_data:
            segment_data[segment_name] = []
        
        segment_data[segment_name].append({
            "Segment": record.segment_value,
            "Total Loans": 0,  # Not stored in ECL table, would need to query loans
            "PD": record.pd_value,
            "LGD": record.lgd_value,
            "EAD": record.ead_value,
            "ECL": record.ecl_value
        })
    
    # Convert to DataFrames
    result_dict = {}
    for segment_name, data_list in segment_data.items():
        df = pd.DataFrame(data_list)
        # Sort by ECL descending
        df = df.sort_values("ECL", ascending=False)
        result_dict[segment_name] = df
    
    return result_dict


def db_loans_to_df(loans: List[Loan]) -> pd.DataFrame:
    """
    Convert list of Loan model instances to DataFrame.
    
    Args:
        loans: List of Loan model instances
        
    Returns:
        DataFrame with loan data (includes derived age_group column)
    """
    if not loans:
        return pd.DataFrame()
    
    data = []
    for loan in loans:
        data.append({
            "loan_id": loan.loan_id,
            "user_id": loan.user_id,
            "loan_amnt": loan.loan_amount,  # DB has loan_amount, DF has loan_amnt
            "loan_intent": loan.loan_intent,
            "loan_int_rate": loan.loan_int_rate,
            "loan_percent_income": loan.loan_percent_income,
            "credit_score": loan.credit_score,
            "person_income": loan.person_income,
            "person_age": loan.person_age,
            "person_gender": loan.person_gender,
            "person_education": loan.person_education,
            "person_emp_exp": loan.person_emp_exp,
            "person_home_ownership": loan.home_ownership,  # DB has home_ownership, DF has person_home_ownership
            "cb_person_cred_hist_length": loan.cb_person_cred_hist_length,
            "previous_loan_defaults_on_file": loan.previous_loan_defaults_on_file,
            "loan_status": loan.loan_status,
            "created_at": loan.created_at
        })
    
    df = pd.DataFrame(data)
    
    # Recreate age_group column from person_age
    df = create_age_groups(df)
    
    return df


def ecl_record_to_segment_dict(ecl_record: ECLSegmentCalculation) -> Dict[str, Any]:
    """
    Convert a single ECL record to segment dictionary for RAG queries.
    
    Args:
        ecl_record: ECLSegmentCalculation model instance
        
    Returns:
        Dictionary with segment information
    """
    return {
        "ecl_id": ecl_record.ecl_id,
        "segment_type": ecl_record.segment_name,
        "segment": ecl_record.segment_value,
        "total_loans": 0,  # Would need to query loans table
        "pd": ecl_record.pd_value,
        "lgd": ecl_record.lgd_value,
        "ead": ecl_record.ead_value,
        "ecl": ecl_record.ecl_value,
        "user_id": ecl_record.user_id,
        "created_at": ecl_record.created_at.isoformat() if ecl_record.created_at else None
    }

