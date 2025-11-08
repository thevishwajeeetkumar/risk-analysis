# Risk Analysis Platform

Cloud-ready ECL analytics with a FastAPI backend, Neon + Pinecone persistence, and a Vite/React dashboard. Upload loan portfolios, compute segment-level ECL, and query insights via RAG—secured with JWT auth and RBAC.

## Live Deployments
- Frontend: https://risk-frontend-snowy.vercel.app
- Backend: https://risk-analysis-3bwh.onrender.com

## Key Features
- **Automated pipeline:** CSV/XLSX ingest → preprocessing → DB schema validation → loan storage → ECL (PD/LGD/EAD) → RAG embedding.
- **Auth & RBAC:** JWT login/register, CRO bypass, analyst auto-grants for safe segments (`loan_intent`, `person_gender`, `person_education`, `person_home_ownership`).
- **RAG insights:** LangChain documents stored in Pinecone; OpenAI powers summarised recommendations.
- **Ops hardening:** Rate limiting, CORS controls, environment-driven config, async SQLAlchemy sessions.

## Architecture & Workflow
```
User → React SPA (Vercel) → FastAPI (Render) → Neon PostgreSQL
                                   ↘ Pinecone ↔ OpenAI (RAG)
```
1. User registers/logs in (`/auth/register`, `/auth/login`).
2. Upload loans to `/api/upload` (100 MB max) → async pipeline persists loans and embeds segments.
3. Query `/api/query` with natural language → RBAC filters segments → verdict, metrics, recommendations returned.
4. CROs manage permissions via `/auth/permissions`; analysts see only authorised segments.

## API Surface
| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/auth/register` | Create user (Analyst/CRO) |
| POST | `/auth/login` | OAuth2 password flow, returns JWT |
| GET | `/auth/me` | Current user profile |
| POST | `/auth/permissions` | Grant permissions (CRO) |
| GET | `/auth/permissions/{user_id}` | View permissions (CRO) |
| POST | `/api/upload` | Upload loan dataset |
| POST | `/api/query` | Query embedded segments |
| GET | `/api/segments` | List stored ECL segments |
| GET | `/` | API metadata |

## Repository Layout
```
server/         FastAPI app (api, auth, core, db, tests)
risk-frontend/  Vite React SPA (hooks, components, pages)
api/            Vercel-compatible entrypoint (optional)
```

## Local Development
Backend
```bash
cd server
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env        # set DATABASE_URL, JWT_SECRET_KEY, etc.
uvicorn api.main:app --reload
```

Frontend
```bash
cd risk-frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
npm run dev
```

## Production Deployment
**Backend (Render)**
- Build command: `pip install -r server/requirements.txt`
- Start command: `cd server && uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Required env vars: `DATABASE_URL`, `JWT_SECRET_KEY`, `OPENAI_API_KEY`, `PINECONE_API_KEY`, `ALLOWED_ORIGINS=https://risk-frontend-snowy.vercel.app`, `RBAC_AUTO_GRANT_SEGMENTS=loan_intent,person_gender,person_education,person_home_ownership`.

**Frontend (Vercel)**
- Env var: `VITE_API_BASE_URL=https://risk-analysis-3bwh.onrender.com`
- Build: `npm run build`, output `dist`.

## Testing & Validation
Backend: `pytest` (pipelines, RBAC, Pinecone flow, auth).  
Frontend: `npm run lint` plus manual smoke (login → upload sample dataset → query).  
Health check: `GET https://risk-analysis-3bwh.onrender.com/vercel/health`.

## Troubleshooting
- **CORS 400:** Ensure `ALLOWED_ORIGINS` includes frontend URL (no trailing slash).
- **RBAC auto-grant failures:** Constraint likely missing segment; update DB or adjust `RBAC_AUTO_GRANT_SEGMENTS`.
- **MissingGreenlet errors:** Deploy latest backend; avoid DB calls after session rollback.
- **Upload errors:** Confirm CSV headers match `core/config.py::REQUIRED_COLUMNS`.

## Roadmap
- Refresh tokens & remember-me.
- Automated end-to-end tests.
- Advanced reporting (PDF exports, alerts).
- Expand RBAC segments once DB constraints updated.

## Credits
- FastAPI/RAG pipeline, schema hardening, documentation: GPT-5 Codex agent.
- Frontend integration & UX wiring: React/Tailwind SPA.
- Infrastructure: Neon, Pinecone, OpenAI, Render, Vercel.

For detailed fix logs and methodologies, see `server/docs/`.


