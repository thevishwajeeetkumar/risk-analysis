"""Test script for file_id filtering in RAG operations."""

import pandas as pd
from core.rag_engine import get_rag_engine


TEST_USERNAME = "test_file_filter_user"
FILE_ID_1 = "test_file_001"
FILE_ID_2 = "test_file_002"


def test_file_id_filtering():
    """Test that file_id filtering works correctly for queries and deletion."""
    print("\n" + "="*60)
    print("TESTING FILE_ID FILTERING")
    print("="*60)
    
    # Initialize RAG engine
    rag_engine = get_rag_engine()
    
    # Clear any existing test data
    print("\n0. Cleaning up any previous test data...")
    try:
        rag_engine.clear_index(TEST_USERNAME)
    except Exception:
        pass
    
    # Create sample ECL data for FILE 1
    sample_ecl_file1 = {
        'loan_intent': pd.DataFrame({
            'Segment': ['VENTURE', 'EDUCATION'],
            'Total Loans': [100, 200],
            'PD': [0.85, 0.25],
            'LGD': [0.40, 0.35],
            'EAD': [12000, 8000],
            'ECL': [4080, 700]
        })
    }
    
    # Create sample ECL data for FILE 2
    sample_ecl_file2 = {
        'loan_intent': pd.DataFrame({
            'Segment': ['PERSONAL', 'MEDICAL'],
            'Total Loans': [150, 180],
            'PD': [0.20, 0.30],
            'LGD': [0.35, 0.40],
            'EAD': [10000, 9000],
            'ECL': [700, 1080]
        })
    }
    
    # Test 1: Upload FILE 1
    print("\n1. Uploading FILE 1 (VENTURE, EDUCATION)...")
    try:
        result1 = rag_engine.embed_segments(
            sample_ecl_file1, 
            FILE_ID_1, 
            username=TEST_USERNAME, 
            user_id=1
        )
        print(f"✅ FILE 1 uploaded: {result1['chunks']} chunks")
    except Exception as e:
        print(f"❌ FILE 1 upload failed: {e}")
        return
    
    # Test 2: Upload FILE 2
    print("\n2. Uploading FILE 2 (PERSONAL, MEDICAL)...")
    try:
        result2 = rag_engine.embed_segments(
            sample_ecl_file2, 
            FILE_ID_2, 
            username=TEST_USERNAME, 
            user_id=1
        )
        print(f"✅ FILE 2 uploaded: {result2['chunks']} chunks")
    except Exception as e:
        print(f"❌ FILE 2 upload failed: {e}")
        return
    
    # Test 3: Query without file_id filter (should find segments from both files)
    print("\n3. Query WITHOUT file_id filter...")
    try:
        result = rag_engine.query_segments(
            "What loan purposes are covered?",
            username=TEST_USERNAME,
            top_k=5
        )
        segments = result['segments']
        print(f"   Retrieved {len(segments)} segments:")
        for seg in segments:
            segment_name = seg.get('segment', 'N/A')
            segment_file = seg.get('file_id', 'N/A')  # This won't show in output but is used internally
            print(f"     - {segment_name}")
        
        # Check if we got segments from both files
        segment_names = {seg['segment'] for seg in segments if seg.get('segment')}
        expected_all = {'VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL'}
        found = segment_names.intersection(expected_all)
        print(f"   Expected segments from both files: {expected_all}")
        print(f"   Found: {found}")
        if len(found) >= 2:
            print("   ✅ Retrieved segments from multiple files")
        else:
            print("   ⚠️ May have retrieved from limited sources")
    except Exception as e:
        print(f"❌ Query without filter failed: {e}")
    
    # Test 4: Query WITH file_id filter for FILE 1
    print(f"\n4. Query WITH file_id filter (FILE 1: {FILE_ID_1})...")
    try:
        result = rag_engine.query_segments(
            "What loan purposes are covered?",
            username=TEST_USERNAME,
            file_id=FILE_ID_1,
            top_k=5
        )
        segments = result['segments']
        print(f"   Retrieved {len(segments)} segments:")
        for seg in segments:
            segment_name = seg.get('segment', 'N/A')
            print(f"     - {segment_name}")
        
        # Verify we only got FILE 1 segments
        segment_names = {seg['segment'] for seg in segments if seg.get('segment')}
        expected_file1 = {'VENTURE', 'EDUCATION'}
        unexpected_file2 = {'PERSONAL', 'MEDICAL'}
        
        if segment_names.issubset(expected_file1):
            print(f"   ✅ All segments are from FILE 1: {segment_names}")
        else:
            unexpected = segment_names.intersection(unexpected_file2)
            if unexpected:
                print(f"   ❌ Found unexpected FILE 2 segments: {unexpected}")
            else:
                print(f"   ⚠️ Segments: {segment_names}")
    except Exception as e:
        print(f"❌ Query with FILE 1 filter failed: {e}")
    
    # Test 5: Query WITH file_id filter for FILE 2
    print(f"\n5. Query WITH file_id filter (FILE 2: {FILE_ID_2})...")
    try:
        result = rag_engine.query_segments(
            "What loan purposes are covered?",
            username=TEST_USERNAME,
            file_id=FILE_ID_2,
            top_k=5
        )
        segments = result['segments']
        print(f"   Retrieved {len(segments)} segments:")
        for seg in segments:
            segment_name = seg.get('segment', 'N/A')
            print(f"     - {segment_name}")
        
        # Verify we only got FILE 2 segments
        segment_names = {seg['segment'] for seg in segments if seg.get('segment')}
        expected_file2 = {'PERSONAL', 'MEDICAL'}
        unexpected_file1 = {'VENTURE', 'EDUCATION'}
        
        if segment_names.issubset(expected_file2):
            print(f"   ✅ All segments are from FILE 2: {segment_names}")
        else:
            unexpected = segment_names.intersection(unexpected_file1)
            if unexpected:
                print(f"   ❌ Found unexpected FILE 1 segments: {unexpected}")
            else:
                print(f"   ⚠️ Segments: {segment_names}")
    except Exception as e:
        print(f"❌ Query with FILE 2 filter failed: {e}")
    
    # Test 6: Delete FILE 1 only
    print(f"\n6. Deleting FILE 1 vectors only ({FILE_ID_1})...")
    try:
        rag_engine.clear_index(TEST_USERNAME, file_id=FILE_ID_1)
        print("   ✅ FILE 1 deletion successful")
    except Exception as e:
        print(f"❌ FILE 1 deletion failed: {e}")
        return
    
    # Test 7: Query after FILE 1 deletion (should only find FILE 2)
    print("\n7. Query after FILE 1 deletion...")
    try:
        result = rag_engine.query_segments(
            "What loan purposes are covered?",
            username=TEST_USERNAME,
            top_k=5
        )
        segments = result['segments']
        print(f"   Retrieved {len(segments)} segments:")
        for seg in segments:
            segment_name = seg.get('segment', 'N/A')
            print(f"     - {segment_name}")
        
        segment_names = {seg['segment'] for seg in segments if seg.get('segment')}
        expected_file2 = {'PERSONAL', 'MEDICAL'}
        unexpected_file1 = {'VENTURE', 'EDUCATION'}
        
        has_file1 = bool(segment_names.intersection(unexpected_file1))
        has_file2 = bool(segment_names.intersection(expected_file2))
        
        if has_file2 and not has_file1:
            print(f"   ✅ Only FILE 2 segments remain: {segment_names}")
        elif has_file1:
            print(f"   ❌ FILE 1 segments still present: {segment_names.intersection(unexpected_file1)}")
        else:
            print(f"   ⚠️ Retrieved segments: {segment_names}")
    except Exception as e:
        print(f"❌ Query after deletion failed: {e}")
    
    # Cleanup
    print("\n8. Final cleanup...")
    try:
        rag_engine.clear_index(TEST_USERNAME)
        print("✅ All test data cleared")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print("\n" + "="*60)
    print("FILE_ID FILTERING TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
    elif not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY not set")
    else:
        test_file_id_filtering()

