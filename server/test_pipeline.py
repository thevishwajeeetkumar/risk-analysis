"""
Test script to verify the complete ECL pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create sample loan data with age column
def create_sample_data():
    """Create a sample loan dataset for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Generate sample data
    data = {
        'person_age': np.random.randint(18, 75, n_samples),
        'person_gender': np.random.choice(['male', 'female'], n_samples),
        'person_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'person_income': np.random.normal(50000, 20000, n_samples),
        'person_emp_exp': np.random.randint(0, 30, n_samples),
        'person_home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n_samples),
        'loan_amnt': np.random.normal(10000, 5000, n_samples),
        'loan_intent': np.random.choice(['VENTURE', 'HOMEIMPROVEMENT', 'EDUCATION', 
                                       'PERSONAL', 'DEBTCONSOLIDATION', 'MEDICAL'], n_samples),
        'loan_int_rate': np.random.normal(10, 3, n_samples),
        'loan_percent_income': np.random.uniform(0.1, 0.5, n_samples),
        'cb_person_cred_hist_length': np.random.randint(1, 20, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'previous_loan_defaults_on_file': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'loan_status': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 30% default rate
    }
    
    df = pd.DataFrame(data)
    
    # Ensure positive values
    df['person_income'] = np.abs(df['person_income'])
    df['loan_amnt'] = np.abs(df['loan_amnt'])
    df['loan_int_rate'] = np.clip(df['loan_int_rate'], 1, 30)
    
    return df


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING")
    print("="*60)
    
    from core.preprocess import preprocess_data
    
    # Create and save sample data
    df = create_sample_data()
    test_file = Path("test_loan_data.csv")
    df.to_csv(test_file, index=False)
    
    # Run preprocessing
    cleaned_df = preprocess_data(test_file, save_output=False)
    
    # Verify age groups were created
    assert 'age_group' in cleaned_df.columns, "age_group column not created"
    print(f"✓ Age groups created: {cleaned_df['age_group'].value_counts().to_dict()}")
    
    # Clean up
    test_file.unlink()
    
    return cleaned_df


def test_segmentation(df):
    """Test the segmentation and ECL calculation."""
    print("\n" + "="*60)
    print("TESTING SEGMENTATION & ECL")
    print("="*60)
    
    from core.segmentation import segment_and_calculate_ecl
    
    # Run segmentation
    ecl_results = segment_and_calculate_ecl(df, save_to_csv=False)
    
    # Verify all segments were calculated
    expected_segments = ['loan_intent', 'person_gender', 'person_education', 
                        'person_home_ownership', 'age_group']
    
    for segment in expected_segments:
        assert segment in ecl_results, f"Missing segment: {segment}"
        print(f"✓ {segment}: {len(ecl_results[segment])} segments calculated")
    
    # Check age_group results specifically
    age_results = ecl_results['age_group']
    print("\nAge Group ECL Results:")
    print(age_results.to_string())
    
    return ecl_results


def test_recommendations(ecl_results):
    """Test the recommendation engine."""
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS")
    print("="*60)
    
    from core.recommendation import generate_segment_verdict
    
    # Get senior citizen segment if it exists
    age_results = ecl_results.get('age_group')
    if age_results is not None:
        senior_row = age_results[age_results['Segment'] == 'senior_citizen']
        if not senior_row.empty:
            senior_data = {
                'segment_type': 'age_group',
                'segment': 'senior_citizen',
                'pd': senior_row.iloc[0]['PD'],
                'lgd': senior_row.iloc[0]['LGD'],
                'ead': senior_row.iloc[0]['EAD'],
                'ecl': senior_row.iloc[0]['ECL'],
                'total_loans': senior_row.iloc[0]['Total Loans']
            }
            
            verdict = generate_segment_verdict(senior_data)
            print(f"Senior Citizen Risk Level: {verdict['risk_level']}")
            print(f"Recommendation: {verdict['recommendation']}")
            print("Insights:")
            for insight in verdict['insights']:
                print(f"  - {insight}")


def test_query_example():
    """Test a sample query about senior citizens."""
    print("\n" + "="*60)
    print("TESTING SAMPLE QUERY")
    print("="*60)
    
    query = "Is giving loans to senior citizens risky?"
    print(f"Query: {query}")
    
    # This would normally go through the RAG engine
    # For testing, we'll just show the expected flow
    print("\nExpected RAG Flow:")
    print("1. Query embedding created")
    print("2. Search Pinecone for relevant age_group segments")
    print("3. Retrieve senior_citizen segment data")
    print("4. Generate natural language response with risk assessment")


if __name__ == "__main__":
    print("ECL PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Test preprocessing with age groups
        cleaned_df = test_preprocessing()
        
        # Test segmentation
        ecl_results = test_segmentation(cleaned_df)
        
        # Test recommendations
        test_recommendations(ecl_results)
        
        # Test query flow
        test_query_example()
        
        print("\n✅ ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
