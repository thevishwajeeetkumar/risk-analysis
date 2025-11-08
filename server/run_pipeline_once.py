"""Manual pipeline runner for testing without HTTP."""
import asyncio
from pathlib import Path

from db.database import async_session_factory
from db import crud
from core.pipeline import process_uploaded_file

CSV_PATH = Path('data/uploads/loan_data.csv')

async def run_pipeline():
    async with async_session_factory() as session:
        users = await crud.list_users(session, limit=1)
        if not users:
            print('[ERROR] No users found in database')
            return
        user = users[0]
        print(f"[INFO] Using user {user.username} (ID: {user.user_id})")
        result = await process_uploaded_file(
            file_path=str(CSV_PATH),
            file_id='manualtest',
            user_id=user.user_id,
            username=user.username,
            db=session
        )
        print('[RESULT]', result)

if __name__ == '__main__':
    asyncio.run(run_pipeline())
