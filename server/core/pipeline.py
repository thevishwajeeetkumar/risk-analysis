"""
Pipeline Orchestrator

Coordinates the full ECL calculation pipeline with database storage:
1. Preprocess uploaded file
2. Store loans in database
3. Segment data and calculate ECL
4. Store ECL results in database
5. Save segment CSVs (for backward compatibility)
6. Embed to Pinecone vector database

This is called by the /api/upload endpoint.
"""

import uuid
from pathlib import Path
from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession

from core.preprocess import preprocess_data
from core.segmentation import segment_and_calculate_ecl, get_segment_statistics
from core.rag_engine import get_rag_engine
from db import crud
from db.transformers import df_to_loan_records, ecl_results_to_db_records


async def process_uploaded_file(
    file_path: str,
    file_id: str,
    user_id: int,
    username: str,
    db: AsyncSession
) -> Dict:
    """
    Complete async pipeline for processing an uploaded loan data file with database storage.
    
    Steps:
    1. Preprocess: Clean data (impute, cap outliers, create age groups)
    2. Store individual loans in database
    3. Segment: Generate segments by loan_intent, gender, education, home_ownership, age_group
    4. Calculate ECL: Compute PD, LGD, EAD, ECL for each segment
    5. Store ECL calculations in database
    6. Save: Write segment CSVs to data/segments/ (backward compatibility)
    7. Embed: Store segments in Pinecone for RAG querying
    
    Args:
        file_path: Path to uploaded CSV/XLSX file
        file_id: Unique identifier for this file
        user_id: ID of the user uploading the file
        username: Username (used for per-user Pinecone index)
        db: Async database session
    
    Returns:
        Dictionary with:
            - file_id: Identifier
            - status: "success" or "error"
            - message: Status message
            - loan_count: Number of loans stored
            - statistics: Summary stats (total_loans, avg_pd, total_ecl, etc.)
            - segments: List of segment types processed
    """
    print("\n" + "="*70)
    print(f"PROCESSING FILE: {Path(file_path).name}")
    print(f"FILE ID: {file_id}")
    print(f"USER ID: {user_id}")
    print(f"USERNAME: {username}")
    print("="*70)
    
    try:
        # Step 1: Preprocess data (remains sync)
        print("\n[STEP 1] Preprocessing...")
        clean_df = preprocess_data(
            input_path=file_path,
            save_output=True,
            output_filename=f"clean_{file_id}.csv"
        )
        
        # Step 2: Store individual loans in database
        print("\n[STEP 2] Storing loans in database...")
        loan_records = df_to_loan_records(clean_df, user_id)
        loan_ids = await crud.bulk_insert_loans(db, loan_records)
        print(f"[SUCCESS] Stored {len(loan_ids)} loans in database")
        
        # Step 3 & 4: Segment and calculate ECL
        print("\n[STEP 3] Segmentation & ECL Calculation...")
        ecl_results = segment_and_calculate_ecl(
            df=clean_df,
            save_to_csv=True,  # Keep CSV export for backward compatibility
            file_id=file_id
        )
        
        # Step 5: Store ECL calculations in database
        print("\n[STEP 4] Storing ECL calculations in database...")
        
        # Ensure ECL table schema is correct (drop CHECK constraints)
        from db.schema_check import ensure_ecl_schema
        await ensure_ecl_schema(db)
        
        ecl_records = ecl_results_to_db_records(ecl_results, user_id)
        ecl_ids = await crud.bulk_insert_ecl_calculations(db, ecl_records)
        print(f"[SUCCESS] Stored {len(ecl_ids)} ECL calculations in database")
        
        # Step 6: Get statistics
        print("\n[STEP 5] Calculating Statistics...")
        statistics = get_segment_statistics(ecl_results)
        
        # Step 7: Embed to Pinecone (now includes user_id)
        print("\n[STEP 6] Embedding to Vector Database...")
        rag_engine = get_rag_engine()
        # Pass user_id to rag_engine for metadata and capture embedding stats
        embedding_stats = rag_engine.embed_segments(
            ecl_results,
            file_id,
            username=username,
            user_id=user_id
        )
        
        print("\n" + "="*70)
        print("[SUCCESS] PIPELINE COMPLETE")
        print("="*70)
        print(f"Total Loans: {len(loan_ids):,}")
        print(f"Total Segments: {statistics.get('total_segments', 0)}")
        print(f"Average PD: {statistics.get('avg_pd', 0):.2%}")
        print(f"Total ECL: ${statistics.get('total_ecl', 0):,.2f}")
        print("="*70)
        
        return {
            "file_id": file_id,
            "status": "success",
            "message": "File processed successfully",
            "loan_count": len(loan_ids),
            "statistics": statistics,
            "segments": list(ecl_results.keys()),
            "embedding": embedding_stats
        }
    
    except Exception as e:
        print(f"\n[ERROR] Error during pipeline: {str(e)}")
        # Rollback is handled by the database session in get_db()
        return {
            "file_id": file_id,
            "status": "error",
            "message": f"Error processing file: {str(e)}",
            "loan_count": 0,
            "statistics": {},
            "segments": []
        }

