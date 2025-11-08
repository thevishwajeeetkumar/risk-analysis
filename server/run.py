"""
Entry point for the ECL Calculation & RAG API

Run with: python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting ECL Calculation & RAG API")
    print("=" * 60)
    print(f"Server will be available at:")
    print(f"  - http://localhost:8000")
    print(f"  - http://127.0.0.1:8000")
    print(f"  - http://0.0.0.0:8000")
    print(f"\nAPI Documentation:")
    print(f"  - Swagger UI: http://localhost:8000/docs")
    print(f"  - ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    print("\nWaiting for requests...\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True  # Enable access logging
    )

