"""
Test Suite for Pinecone Field Map Integration

Tests the fix for the "field_map is required in embed" error by validating:
1. ECL DataFrame structure validation
2. Document creation from valid ECL results
3. Pinecone index creation with proper field_map
4. Error handling for malformed data
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from core.rag_engine import ECLRagEngine


@pytest.fixture
def valid_ecl_dataframe():
    """Create a valid ECL DataFrame with all required columns."""
    return pd.DataFrame({
        'Segment': ['EDUCATION', 'MEDICAL', 'VENTURE'],
        'Total Loans': [50, 75, 25],
        'PD': [0.15, 0.10, 0.35],
        'LGD': [0.40, 0.38, 0.45],
        'EAD': [15000.00, 20000.00, 12000.00],
        'ECL': [900.00, 760.00, 1890.00]
    })


@pytest.fixture
def invalid_ecl_dataframe_missing_columns():
    """Create an invalid ECL DataFrame missing required columns."""
    return pd.DataFrame({
        'Segment': ['EDUCATION', 'MEDICAL'],
        'Total Loans': [50, 75],
        'PD': [0.15, 0.10],
        # Missing: LGD, EAD, ECL
    })


@pytest.fixture
def ecl_dataframe_with_nan():
    """Create ECL DataFrame with NaN values to test sanitization."""
    return pd.DataFrame({
        'Segment': ['EDUCATION', None, 'VENTURE'],
        'Total Loans': [50, 75, pd.NA],
        'PD': [0.15, float('nan'), 0.35],
        'LGD': [0.40, 0.38, None],
        'EAD': [15000.00, pd.NA, 12000.00],
        'ECL': [900.00, 760.00, float('nan')]
    })


@pytest.fixture
def ecl_results_valid():
    """Create a valid ecl_results dictionary."""
    return {
        'loan_intent': pd.DataFrame({
            'Segment': ['EDUCATION', 'MEDICAL', 'VENTURE'],
            'Total Loans': [50, 75, 25],
            'PD': [0.15, 0.10, 0.35],
            'LGD': [0.40, 0.38, 0.45],
            'EAD': [15000.00, 20000.00, 12000.00],
            'ECL': [900.00, 760.00, 1890.00]
        }),
        'person_gender': pd.DataFrame({
            'Segment': ['male', 'female'],
            'Total Loans': [80, 70],
            'PD': [0.18, 0.12],
            'LGD': [0.42, 0.39],
            'EAD': [16000.00, 18000.00],
            'ECL': [1209.60, 842.40]
        })
    }


class TestECLDataFrameValidation:
    """Test ECL DataFrame validation logic."""
    
    def test_validate_valid_dataframe(self, valid_ecl_dataframe):
        """Test validation passes for valid DataFrame."""
        engine = ECLRagEngine()
        result = engine._validate_ecl_dataframe(valid_ecl_dataframe, 'loan_intent')
        assert result is True
    
    def test_validate_missing_columns(self, invalid_ecl_dataframe_missing_columns):
        """Test validation fails for DataFrame missing required columns."""
        engine = ECLRagEngine()
        result = engine._validate_ecl_dataframe(invalid_ecl_dataframe_missing_columns, 'loan_intent')
        assert result is False
    
    def test_validate_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        engine = ECLRagEngine()
        empty_df = pd.DataFrame()
        result = engine._validate_ecl_dataframe(empty_df, 'test_segment')
        assert result is False


class TestDocumentCreation:
    """Test document creation from ECL results."""
    
    def test_create_documents_from_valid_ecl(self, ecl_results_valid):
        """Test successful document creation from valid ECL results."""
        engine = ECLRagEngine()
        file_id = "test_file_123"
        user_id = 42
        
        documents = engine._create_segment_documents(ecl_results_valid, file_id, user_id)
        
        # Should create 5 documents (3 from loan_intent + 2 from person_gender)
        assert len(documents) == 5
        
        # Check first document structure
        doc = documents[0]
        assert doc.page_content is not None
        assert 'Segment Type:' in doc.page_content
        assert doc.metadata['file_id'] == file_id
        assert doc.metadata['user_id'] == user_id
        assert 'segment_type' in doc.metadata
        assert 'segment' in doc.metadata
        assert 'pd' in doc.metadata
        assert 'lgd' in doc.metadata
        assert 'ead' in doc.metadata
        assert 'ecl' in doc.metadata
    
    def test_create_documents_handles_nan(self, ecl_dataframe_with_nan):
        """Test document creation handles NaN values gracefully."""
        engine = ECLRagEngine()
        ecl_results = {'test_segment': ecl_dataframe_with_nan}
        file_id = "test_nan_123"
        
        # Should not raise error, should sanitize NaN values
        documents = engine._create_segment_documents(ecl_results, file_id)
        
        assert len(documents) > 0
        # Check that NaN values were replaced
        for doc in documents:
            assert not any(pd.isna(v) for v in doc.metadata.values())
    
    def test_create_documents_with_invalid_segment(self):
        """Test document creation fails gracefully with invalid segment data."""
        engine = ECLRagEngine()
        invalid_ecl_results = {
            'bad_segment': pd.DataFrame({
                'Wrong': ['data'],
                'Columns': ['here']
            })
        }
        file_id = "test_invalid_123"
        
        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError, match="No valid documents created"):
            engine._create_segment_documents(invalid_ecl_results, file_id)
    
    def test_create_documents_empty_ecl_results(self):
        """Test document creation with empty ECL results."""
        engine = ECLRagEngine()
        empty_ecl_results = {}
        file_id = "test_empty_123"
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            engine._create_segment_documents(empty_ecl_results, file_id)


class TestMetadataSanitization:
    """Test metadata sanitization logic."""
    
    def test_sanitize_none_values(self):
        """Test None values are sanitized."""
        engine = ECLRagEngine()
        metadata = {
            'key1': None,
            'key2': 'valid',
            'key3': None
        }
        result = engine._sanitize_metadata(metadata)
        
        assert result['key1'] == ''
        assert result['key2'] == 'valid'
        assert result['key3'] == ''
    
    def test_sanitize_nan_values(self):
        """Test NaN values are sanitized."""
        engine = ECLRagEngine()
        metadata = {
            'float_val': float('nan'),
            'pd_val': 0.15,
            'int_val': 42
        }
        result = engine._sanitize_metadata(metadata)
        
        assert result['float_val'] == 0.0
        assert result['pd_val'] == 0.15
        assert result['int_val'] == 42
    
    def test_sanitize_mixed_types(self):
        """Test sanitization handles various types."""
        engine = ECLRagEngine()
        metadata = {
            'str': 'text',
            'int': 123,
            'float': 45.67,
            'bool': True,
            'none': None,
            'nan': float('nan')
        }
        result = engine._sanitize_metadata(metadata)
        
        assert isinstance(result['str'], str)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['bool'], bool)
        assert result['none'] == ''
        assert result['nan'] == 0.0


@patch('core.rag_engine.pc')
class TestPineconeIndexCreation:
    """Test Pinecone serverless index creation without embed model."""
    
    def test_serverless_index_creation_without_embed(self, mock_pc):
        """Test that serverless index creation uses create_index without embed parameter."""
        from core.rag_engine import get_user_index
        from pinecone import ServerlessSpec
        
        # Mock Pinecone client
        mock_pc.list_indexes.return_value = []
        mock_index = Mock()
        mock_index._index_name = 'test-user'
        mock_pc.Index.return_value = mock_index
        mock_pc.create_index = Mock()
        
        # Call get_user_index
        result = get_user_index('test_user')
        
        # Verify create_index was called (not create_index_for_model)
        mock_pc.create_index.assert_called_once()
        call_args = mock_pc.create_index.call_args
        
        # Verify NO embed parameter is passed
        assert 'embed' not in call_args.kwargs
        
        # Verify required parameters for serverless
        assert call_args.kwargs['name'] == 'test-user'
        assert call_args.kwargs['dimension'] == 1536
        assert call_args.kwargs['metric'] == 'cosine'
        assert 'spec' in call_args.kwargs
    
    def test_index_creation_error_handling(self, mock_pc):
        """Test error handling for INVALID_ARGUMENT errors."""
        from core.rag_engine import get_user_index
        from fastapi import HTTPException
        
        # Mock Pinecone client
        mock_pc.list_indexes.return_value = []
        
        # Simulate INVALID_ARGUMENT error
        mock_pc.create_index = Mock(
            side_effect=Exception("INVALID_ARGUMENT: Model text-embedding-3-small not found")
        )
        
        # Call get_user_index - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            get_user_index('test_user')
        
        # Verify error details
        assert exc_info.value.status_code == 503
        assert "model" in str(exc_info.value.detail).lower()
    
    def test_index_dimension_mismatch_handling(self, mock_pc):
        """Test error handling for dimension mismatches."""
        from core.rag_engine import get_user_index
        from fastapi import HTTPException
        
        # Mock Pinecone client
        mock_pc.list_indexes.return_value = []
        
        # Simulate dimension error
        mock_pc.create_index = Mock(
            side_effect=Exception("INVALID_ARGUMENT: dimension must be between 1 and 20000")
        )
        
        # Call get_user_index - should raise HTTPException with helpful message
        with pytest.raises(HTTPException) as exc_info:
            get_user_index('test_user')
        
        # Verify error is caught and wrapped
        assert exc_info.value.status_code == 503


class TestEmbedSegmentsValidation:
    """Test embed_segments input validation."""
    
    @patch('core.rag_engine.get_user_index')
    def test_embed_empty_ecl_results(self, mock_index):
        """Test embed_segments raises error for empty ECL results."""
        engine = ECLRagEngine()
        
        with pytest.raises(ValueError, match="ECL results dictionary is empty"):
            engine.embed_segments(
                ecl_results={},
                file_id='test_123',
                username='testuser',
                user_id=1
            )
    
    @patch('core.rag_engine.get_user_index')
    def test_embed_invalid_ecl_structure(self, mock_index):
        """Test embed_segments handles invalid ECL structure."""
        engine = ECLRagEngine()
        invalid_ecl = {
            'bad_segment': pd.DataFrame({'wrong': ['columns']})
        }
        
        with pytest.raises(ValueError, match="No valid documents created"):
            engine.embed_segments(
                ecl_results=invalid_ecl,
                file_id='test_123',
                username='testuser',
                user_id=1
            )
    
    @patch('core.rag_engine.get_user_index')
    def test_dimension_mismatch_detection(self, mock_get_index, ecl_results_valid):
        """Test that dimension mismatches between index and embeddings are detected."""
        from fastapi import HTTPException
        engine = ECLRagEngine()
        
        # Mock index with wrong dimension
        mock_index = Mock()
        mock_index._index_name = 'testuser'
        mock_index.describe_index_stats.return_value = {
            'dimension': 768,  # Wrong dimension (should be 1536)
            'total_vector_count': 0
        }
        mock_get_index.return_value = mock_index
        
        # Execute embedding - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.embed_segments(
                ecl_results=ecl_results_valid,
                file_id='test_123',
                username='testuser',
                user_id=1
            )
        
        # Verify error details
        assert exc_info.value.status_code == 400
        assert "dimension mismatch" in str(exc_info.value.detail).lower()
        assert "768" in str(exc_info.value.detail)
        assert "1536" in str(exc_info.value.detail)


class TestEndToEndEmbeddingFlow:
    """Test the complete embedding flow."""
    
    @patch('core.rag_engine.PineconeVectorStore')
    @patch('core.rag_engine.get_user_index')
    def test_successful_embedding_flow(self, mock_get_index, mock_vectorstore, ecl_results_valid):
        """Test complete successful embedding flow with dimension validation."""
        engine = ECLRagEngine()
        
        # Mock Pinecone index with correct dimension
        mock_index = Mock()
        mock_index._index_name = 'testuser'
        mock_index.describe_index_stats.return_value = {
            'dimension': 1536,  # Correct dimension for text-embedding-3-small
            'total_vector_count': 0
        }
        mock_index.fetch.return_value = {'vectors': {}}
        mock_get_index.return_value = mock_index
        
        # Mock vector store
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        
        # Execute embedding
        result = engine.embed_segments(
            ecl_results=ecl_results_valid,
            file_id='test_file_123',
            username='testuser',
            user_id=42
        )
        
        # Verify result structure
        assert result['status'] == 'success'
        assert result['documents'] == 5  # 3 + 2 from fixtures
        assert result['chunks'] > 0
        assert result['vectors_added'] > 0
        
        # Verify dimension validation was called
        mock_index.describe_index_stats.assert_called_once()
        
        # Verify vector store was called
        mock_vs_instance.add_documents.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

