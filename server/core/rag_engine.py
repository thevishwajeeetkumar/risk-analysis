"""RAG (Retrieval-Augmented Generation) engine for ECL segments.

Provides a simplified single-level retriever backed by Pinecone and
OpenAI embeddings. Documents are chunked with one splitter and stored
directly in the vector index (no parent-child persistence layer).
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
import math

from fastapi import HTTPException, status

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # pip install langchain-openai
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pip install langchain-text-splitters

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Pinecone imports
from langchain_pinecone import PineconeVectorStore  # pip install langchain-pinecone :contentReference[oaicite:18]{index=18}
from pinecone import Pinecone, ServerlessSpec  # pip install "pinecone>=3" :contentReference[oaicite:19]{index=19}

from core.config import (
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K
)

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)


# Initialise Pinecone client once
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client initialized successfully")
        print("   Indexes will be created per-user on first request")
    except Exception as e:
        print(f"⚠️  Warning: Pinecone initialization failed: {e}")
        print("   Pinecone features will not be available.")
        pc = None
else:
    print("⚠️  Warning: Pinecone not configured (PINECONE_API_KEY not set)")
    pc = None


def _sanitize_index_name(username: str) -> str:
    index_name = username.lower().replace("_", "-").replace(" ", "-")
    index_name = ''.join(c for c in index_name if c.isalnum() or c == '-')
    index_name = index_name.strip('-')
    return index_name[:45] or "default"


def get_user_index(username: str):
    """
    Create/retrieve a Pinecone index dedicated to the given username.
    
    Uses serverless spec without embed model configuration to comply with
    Pinecone 2025-04 API requirements. OpenAI embeddings are generated
    client-side and upserted as raw vectors.
    
    Args:
        username: Username to create/retrieve index for
        
    Returns:
        Pinecone Index object
        
    Raises:
        HTTPException: If Pinecone is unavailable or index creation fails
    """
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )

    index_name = _sanitize_index_name(username)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"📝 Creating new Pinecone serverless index for user: {username}")
        print(f"   Index name: {index_name}")
        print(f"   Dimension: {PINECONE_DIMENSION} (OpenAI text-embedding-3-small)")
        print(f"   Metric: {PINECONE_METRIC}")
        print(f"   Cloud: {PINECONE_CLOUD}, Region: {PINECONE_REGION}")
        
        try:
            # Create serverless index without embed model configuration
            # OpenAI embeddings are generated client-side via LangChain
            pc.create_index(
                name=index_name,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
            print(f"✅ Index '{index_name}' created successfully")
            print("   Note: First request may take ~60 seconds while index initializes")
            
            # Create Index object and attach name for logging
            index = pc.Index(index_name)
            index._index_name = index_name
            return index
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Failed to create Pinecone index: {error_msg}")
            
            # Provide detailed error guidance based on error type
            if "INVALID_ARGUMENT" in error_msg:
                if "dimension" in error_msg.lower():
                    detail = (
                        f"Pinecone index creation failed due to dimension mismatch. "
                        f"Expected: {PINECONE_DIMENSION} (matching {OPENAI_EMBEDDING_MODEL}). "
                        f"Error: {error_msg}"
                    )
                elif "model" in error_msg.lower():
                    detail = (
                        f"Pinecone serverless does not support the embed.model parameter. "
                        f"This system uses client-side OpenAI embeddings ({OPENAI_EMBEDDING_MODEL}). "
                        f"Error: {error_msg}"
                    )
                else:
                    detail = f"Pinecone index creation failed with invalid argument: {error_msg}"
            elif "ALREADY_EXISTS" in error_msg:
                detail = f"Index '{index_name}' already exists but was not detected. Retrying..."
                print(f"[WARNING] {detail}")
                # Index might have been created by another process, continue
                return pc.Index(index_name)
            else:
                detail = f"Failed to create Pinecone index '{index_name}': {error_msg}"
            
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail
            )
    else:
        print(f"♻️  Using existing Pinecone index: {index_name}")

    # Create Index object and attach name for logging
    index = pc.Index(index_name)
    index._index_name = index_name  # Store name for logging purposes
    return index


class ECLRagEngine:
    """
    RAG Engine for querying ECL segment data using LangChain and Pinecone.
    """
    
    def __init__(self):
        """Initialize RAG engine with Pinecone and OpenAI."""
        self.model = ChatOpenAI(model=OPENAI_MODEL)
        self.embedding = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        self.parser = StrOutputParser()
        
        # Single text splitter for chunking documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP
        )
    
    def _validate_ecl_dataframe(self, ecl_df: pd.DataFrame, segment_type: str) -> bool:
        """
        Validate that ECL DataFrame has required columns.
        
        Args:
            ecl_df: DataFrame to validate
            segment_type: Type of segment (for logging)
        
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['Segment', 'Total Loans', 'PD', 'LGD', 'EAD', 'ECL']
        missing_columns = [col for col in required_columns if col not in ecl_df.columns]
        
        if missing_columns:
            print(f"[ERROR] Segment '{segment_type}' missing required columns: {missing_columns}")
            print(f"[ERROR] Available columns: {list(ecl_df.columns)}")
            return False
        
        return True
    
    def _create_segment_documents(
        self,
        ecl_results: Dict[str, pd.DataFrame],
        file_id: str,
        user_id: Optional[int] = None
    ) -> List[Document]:
        """
        Convert ECL segment DataFrames into LangChain Documents.
        
        Args:
            ecl_results: Dictionary mapping segment_type to ECL DataFrame
            file_id: Identifier for the uploaded file
            user_id: Optional ID of the user who uploaded the file
        
        Returns:
            List of Document objects with ECL segment information
        """
        documents = []
        skipped_count = 0
        validation_errors = []
        
        for segment_type, ecl_df in ecl_results.items():
            # Validate DataFrame structure
            if not self._validate_ecl_dataframe(ecl_df, segment_type):
                validation_errors.append(segment_type)
                continue
            
            for idx, row in ecl_df.iterrows():
                # Safely extract values with defaults using .get() and pd.to_numeric()
                segment = row.get('Segment', 'UNKNOWN')
                if pd.isna(segment) or str(segment).strip() == '':
                    segment = 'UNKNOWN'
                
                total_loans = pd.to_numeric(row.get('Total Loans', 0), errors='coerce')
                if pd.isna(total_loans):
                    total_loans = 0
                
                pd_val = pd.to_numeric(row.get('PD', 0.0), errors='coerce')
                if pd.isna(pd_val):
                    pd_val = 0.0
                
                lgd_val = pd.to_numeric(row.get('LGD', 0.0), errors='coerce')
                if pd.isna(lgd_val):
                    lgd_val = 0.0
                
                ead_val = pd.to_numeric(row.get('EAD', 0.0), errors='coerce')
                if pd.isna(ead_val):
                    ead_val = 0.0
                
                ecl_val = pd.to_numeric(row.get('ECL', 0.0), errors='coerce')
                if pd.isna(ecl_val):
                    ecl_val = 0.0
                
                # Create rich text content
                content = (
                    f"Segment Type: {segment_type}\n"
                    f"Segment: {segment}\n"
                    f"Total Loans: {int(total_loans)}\n"
                    f"Probability of Default (PD): {pd_val:.4f}\n"
                    f"Loss Given Default (LGD): {lgd_val:.4f}\n"
                    f"Exposure at Default (EAD): ${ead_val:.2f}\n"
                    f"Expected Credit Loss (ECL): ${ecl_val:.2f}\n"
                    f"\n"
                    f"Risk Assessment: "
                )
                
                # Add risk interpretation
                if pd_val >= 0.25:
                    content += "HIGH RISK - Consider reducing disbursements or increasing interest rates."
                elif pd_val >= 0.15:
                    content += "MEDIUM RISK - Monitor closely and adjust terms if needed."
                else:
                    content += "LOW RISK - Standard lending terms appropriate."
                
                # Skip segments with empty or whitespace-only content
                if not content or content.strip() == '':
                    print(f"[SKIP] Segment at index {idx} in {segment_type} has empty content")
                    skipped_count += 1
                    continue
                
                # Create document with metadata
                metadata = {
                    "file_id": file_id,
                    "segment_type": segment_type,
                    "segment": str(segment),
                    "total_loans": int(total_loans),
                    "pd": float(pd_val),
                    "lgd": float(lgd_val),
                    "ead": float(ead_val),
                    "ecl": float(ecl_val)
                }
                
                # Add user_id if provided
                if user_id is not None:
                    metadata["user_id"] = user_id
                
                # Sanitize metadata to ensure JSON-serializable primitives and no NaN/None
                metadata = self._sanitize_metadata(metadata)
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
        
        if skipped_count > 0:
            print(f"[INFO] Skipped {skipped_count} segment(s) with empty content")
        
        if validation_errors:
            print(f"[WARNING] Failed to process {len(validation_errors)} segment type(s): {validation_errors}")
            print(f"[INFO] Successfully created {len(documents)} documents from valid segments")
        
        if not documents:
            raise ValueError(
                f"No valid documents created. Validation errors: {validation_errors}. "
                "Please ensure ECL calculation produced valid segment data with columns: "
                "Segment, Total Loans, PD, LGD, EAD, ECL"
            )
        
        return documents
    
    def embed_segments(
        self,
        ecl_results: Dict[str, pd.DataFrame],
        file_id: str,
        username: str,
        user_id: Optional[int] = None
    ) -> dict:
        """
        Embed ECL segments into the Pinecone index dedicated to the given username.
        Uses deterministic chunk IDs and checks for existing vectors to ensure idempotency.

        Args:
            ecl_results: Dictionary from segment_and_calculate_ecl()
            file_id: Identifier for the uploaded file
            username: Username used to resolve the Pinecone index
            user_id: Optional numeric identifier stored in metadata
        
        Returns:
            Dictionary with embedding statistics and status
        
        Raises:
            ValueError: If no valid documents can be created from ECL results
            HTTPException: If Pinecone service is unavailable
        """
        print("\n" + "="*60)
        print("EMBEDDING SEGMENTS TO PINECONE")
        if user_id is not None:
            print(f"User ID: {user_id}")
        print(f"Username: {username}")
        print(f"File ID: {file_id}")
        print("="*60)
        
        # Validate input
        if not ecl_results:
            raise ValueError("ECL results dictionary is empty. No segments to embed.")
        
        print(f"[INFO] Processing {len(ecl_results)} segment type(s): {list(ecl_results.keys())}")
        
        # Create documents from segments (with validation)
        try:
            documents = self._create_segment_documents(ecl_results, file_id, user_id)
            print(f"[SUCCESS] Created {len(documents)} segment documents")
        except ValueError as e:
            print(f"[ERROR] Failed to create documents: {str(e)}")
            raise
        
        # Add unique IDs to documents for tracking
        for doc in documents:
            # Generate deterministic doc_id: file_id + segment_type + hash(segment + content snippet)
            content_snippet = doc.page_content[:100] if doc.page_content else ""
            composite_key = f"{file_id}:{doc.metadata['segment_type']}:{doc.metadata['segment']}:{content_snippet}"
            doc_hash = hashlib.sha256(composite_key.encode('utf-8')).hexdigest()[:16]
            doc_id = f"{file_id}:{doc.metadata['segment_type']}:{doc_hash}"
            doc.metadata['doc_id'] = doc_id

        # Single-level chunking
        split_docs = self.splitter.split_documents(documents)

        # Generate deterministic chunk IDs based on doc_id and chunk index
        ids = []
        for idx, doc in enumerate(split_docs):
            chunk_id = f"{doc.metadata['doc_id']}_chunk_{idx}"
            doc.metadata['chunk_id'] = chunk_id
            ids.append(chunk_id)

        if split_docs:
            try:
                print(f"[INFO] Getting/creating Pinecone index for user: {username}")
                index = get_user_index(username)
                print(f"[SUCCESS] Index ready: {index._index_name}")
                
                # Validate index dimension matches embedding model
                try:
                    index_stats = index.describe_index_stats()
                    index_dimension = index_stats.get('dimension')
                    if index_dimension and index_dimension != PINECONE_DIMENSION:
                        error_msg = (
                            f"Dimension mismatch: Index '{index._index_name}' has dimension {index_dimension}, "
                            f"but {OPENAI_EMBEDDING_MODEL} produces {PINECONE_DIMENSION}-dimensional vectors. "
                            f"Please delete the index or use a different username."
                        )
                        print(f"[ERROR] {error_msg}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=error_msg
                        )
                    print(f"[VALIDATION] Index dimension ({index_dimension}) matches embedding model ({PINECONE_DIMENSION})")
                except HTTPException:
                    raise
                except Exception as e:
                    # Non-critical validation error, log and continue
                    print(f"[WARNING] Could not validate index dimension: {e}")
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"[ERROR] Failed to get/create Pinecone index: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to initialize Pinecone index: {str(e)}"
                )
            
            # Probe index to check if vectors already exist (idempotency check)
            # Sample a subset of IDs to check (up to 10 for efficiency)
            sample_ids = ids[:min(10, len(ids))]
            try:
                fetch_result = index.fetch(ids=sample_ids)
                existing_count = len(fetch_result.get('vectors', {}))
                
                if existing_count > 0:
                    print(f"[IDEMPOTENCY] Found {existing_count} existing vectors (out of {len(sample_ids)} sampled)")
                    print(f"[IDEMPOTENCY] Skipping re-upsert to maintain idempotency")
                    return {
                        "documents": len(documents),
                        "chunks": len(split_docs),
                        "vectors_added": 0,
                        "status": "skipped",
                        "reason": "vectors_already_exist"
                    }
                else:
                    print(f"[IDEMPOTENCY] No existing vectors found, proceeding with upsert")
            except Exception as e:
                print(f"[WARNING] Error checking for existing vectors: {e}")
                print("[INFO] Proceeding with upsert")
            
            # Proceed with upsert
            try:
                vector_store = PineconeVectorStore(index=index, embedding=self.embedding)
                print(f"[INFO] Embedding {len(split_docs)} chunks from {len(documents)} documents into index '{index._index_name}'...")
                vector_store.add_documents(documents=split_docs, ids=ids)
                print(f"[SUCCESS] Embedded {len(split_docs)} chunks to Pinecone")
            except Exception as e:
                print(f"[ERROR] Failed to embed documents to Pinecone: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to embed segments: {str(e)}"
                )
            
            return {
                "documents": len(documents),
                "chunks": len(split_docs),
                "vectors_added": len(split_docs),
                "status": "success"
            }
        else:
            print("[WARNING] No documents available for embedding")
            return {
                "documents": 0,
                "chunks": 0,
                "vectors_added": 0,
                "status": "no_documents"
            }

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Ensure metadata contains only JSON-serializable primitives and no None/NaN."""
        def sanitize_value(value):
            try:
                # Replace NaN/None
                if value is None:
                    return ""
                if isinstance(value, float) and math.isnan(value):
                    return 0.0
                # Primitive pass-through
                if isinstance(value, (str, int, float, bool)):
                    return value
                # Fallback to string representation
                return str(value)
            except Exception:
                return ""
        return {k: sanitize_value(v) for k, v in metadata.items()}
    
    def query_segments(
        self,
        query: str,
        username: str,
        file_id: Optional[str] = None,
        top_k: int = RAG_TOP_K
    ) -> dict:
        """
        Query the Pinecone index corresponding to `username` for segment insights.

        Args:
            query: Natural language question about ECL segments
            username: Username whose index is queried
            file_id: Optional filter to search only within a specific file's segments
            top_k: Number of segments to retrieve
        """
        print(f"\n[SEARCH] Querying: {query}")
        if file_id:
            print(f"[FILTER] Filtering results by file_id: {file_id}")
        
        index = get_user_index(username)
        vector_store = PineconeVectorStore(index=index, embedding=self.embedding)

        # Create retriever for searching chunks with optional file_id filter
        search_kwargs = {"k": top_k}
        if file_id:
            search_kwargs["filter"] = {"file_id": file_id}
        
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
        # Define the prompt template
        prompt = PromptTemplate(
            template="""You are an expert financial analyst specializing in credit risk and Expected Credit Loss (ECL) analysis.

Based on the following ECL segment data, answer the user's question with specific insights and recommendations.

Context (ECL Segment Data):
{context}

Question: {question}

Provide a detailed answer with:
1. Direct answer to the question
2. Specific segments and their metrics (PD, LGD, EAD, ECL)
3. Risk assessment and business recommendations
4. Any relevant comparisons between segments

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Retrieve documents once
        retrieved_docs = retriever.invoke(query)
        
        # Guardrail: Warn if retrieval is sparse
        if len(retrieved_docs) < 3:
            logger.warning(
                f"[RAG] Sparse retrieval: Only {len(retrieved_docs)} documents retrieved for query '{query[:50]}...'. "
                f"Consider: (1) broader query terms, (2) checking embeddings exist, (3) verifying file_id filter."
            )
        else:
            logger.info(f"[RAG] Retrieved {len(retrieved_docs)} documents successfully")

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        context_text = format_docs(retrieved_docs)

        try:
            rendered_prompt = prompt.format(context=context_text, question=query)
            model_response = self.model.invoke(rendered_prompt)
            answer = self.parser.invoke(model_response)

            # Extract segment information
            segments = []
            for doc in retrieved_docs:
                segments.append({
                    "segment_type": doc.metadata.get("segment_type"),
                    "segment": doc.metadata.get("segment"),
                    "total_loans": doc.metadata.get("total_loans"),
                    "pd": doc.metadata.get("pd"),
                    "lgd": doc.metadata.get("lgd"),
                    "ead": doc.metadata.get("ead"),
                    "ecl": doc.metadata.get("ecl")
                })
            
            return {
                "query": query,
                "answer": answer,
                "segments": segments,
                "context": context_text,
                "retrieval_count": len(retrieved_docs)  # Add retrieval count for visibility
            }
        
        except Exception as e:
            print(f"Error querying segments: {e}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "segments": [],
                "context": ""
            }
    
    def clear_index(self, username: str, file_id: Optional[str] = None):
        """
        Clear a user's Pinecone index or delete vectors for a specific file_id.
        
        Args:
            username: The user whose index should be cleared
            file_id: If provided, only delete vectors for this file_id using metadata filtering
        """
        index = get_user_index(username)
        
        if file_id:
            # Delete vectors filtered by file_id metadata
            try:
                print(f"[INFO] Deleting vectors for file_id: {file_id} from index: {index._index_name}")
                # Pinecone delete with metadata filter
                index.delete(filter={"file_id": file_id})
                print(f"[SUCCESS] Deleted all vectors for file_id: {file_id}")
            except Exception as e:
                print(f"[ERROR] Failed to delete vectors for file_id {file_id}: {e}")
                raise
        else:
            # Delete all vectors in the index
            index.delete(delete_all=True)
            print(f"[SUCCESS] Cleared all vectors from index: {index._index_name}")


# Singleton instance
_rag_engine_instance = None


def get_rag_engine() -> ECLRagEngine:
    """
    Get or create singleton RAG engine instance.
    
    Returns:
        ECLRagEngine instance
    """
    global _rag_engine_instance
    
    if _rag_engine_instance is None:
        _rag_engine_instance = ECLRagEngine()
    
    return _rag_engine_instance

