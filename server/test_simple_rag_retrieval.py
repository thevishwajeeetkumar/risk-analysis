"""Simple test script for the per-user RAG pipeline implementation."""

import pandas as pd
from core.rag_engine import get_rag_engine


TEST_USERNAME = "testuser"


def test_simple_rag_retrieval():
    """Smoke-test the simplified RAG embedding and retrieval flow."""
    print("\n" + "="*60)
    print("TESTING SIMPLE RAG RETRIEVAL")
    print("="*60)
    
    # Initialize RAG engine
    rag_engine = get_rag_engine()
    
    # Create sample ECL data
    sample_ecl_results = {
        'age_group': pd.DataFrame({
            'Segment': ['young', 'middle_aged', 'senior_citizen'],
            'Total Loans': [284, 379, 337],
            'PD': [0.3028, 0.3113, 0.2760],
            'LGD': [0.355, 0.355, 0.355],
            'EAD': [9979.18, 10506.08, 9548.56],
            'ECL': [1072.64, 1161.17, 935.44]
        }),
        'loan_intent': pd.DataFrame({
            'Segment': ['VENTURE', 'EDUCATION', 'PERSONAL'],
            'Total Loans': [100, 200, 150],
            'PD': [0.35, 0.25, 0.20],
            'LGD': [0.40, 0.35, 0.35],
            'EAD': [12000, 8000, 10000],
            'ECL': [1680, 700, 700]
        })
    }
    
    # Test embedding
    print("\n1. Testing Document Embedding...")
    try:
        rag_engine.clear_index(TEST_USERNAME)
    except Exception:
        pass

    try:
        rag_engine.embed_segments(sample_ecl_results, "test_001", username=TEST_USERNAME, user_id=1)
        print("✅ Embedding successful")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return
    
    # Test retrieval
    print("\n2. Testing Simple RAG Retrieval...")
    test_queries = [
        "What is the risk for senior citizens?",
        "Which loan intent has the highest ECL?",
        "Tell me about age groups and their default rates"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        try:
            result = rag_engine.query_segments(query, username=TEST_USERNAME, top_k=3)
            
            # Show retrieved segments
            print(f"Retrieved {len(result['segments'])} segments:")
            for seg in result['segments']:
                print(f"  - {seg['segment_type']}: {seg['segment']} (PD={seg['pd']:.2%}, ECL=${seg['ecl']:.2f})")
            
            # Show answer preview
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"\nAnswer: {answer_preview}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Clear test data
    print("\n3. Cleaning up...")
    try:
        rag_engine.clear_index(TEST_USERNAME)
        print("✅ Test data cleared")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

if __name__ == "__main__":
    # Check for API keys
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
    elif not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY not set")
    else:
        test_simple_rag_retrieval()

