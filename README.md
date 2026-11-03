# KYC MVP

Login → upload NIC/passport + utility bill + typed address → guided live face capture → automatic checks → decision.

```
KYC/
├── backend/            FastAPI (Python 3.12)
│   ├── app/
│   │   ├── llm/        LLM adapters copied unchanged from el-aiee (src/llm)
│   │   ├── routers/    auth.py (login), kyc.py (flow)
│   │   └── services/   llm_service, document_ai, face, id_rules, matching, decision, pipeline
│   ├── models/         YuNet + SFace face models (ONNX, CPU)
│   ├── .env            your settings (adapter, API keys, auth secret) — not committed
│   └── Dockerfile
├── frontend/           Next.js 16 (App Router)
│   ├── app/            /login, /kyc, /api/[...path] (proxy to the backend)
│   ├── components/     DocumentsStep, FaceCaptureStep, ResultStep
│   └── Dockerfile
└── docker-compose.yml  redis + backend + frontend
```

## Run it

```bash
sh backend/scripts/init_env.sh      # creates backend/.env with a new auth secret + file-encryption key
# edit backend/.env: pick LLM_PROVIDER and add that provider's key, e.g.
#   LLM_PROVIDER=gemini
#   GEMINI_API_KEY=...
docker compose up --build
```

- App: http://localhost:3000
- API docs: http://localhost:8000/docs

### Local development (without Docker)

```bash
redis-server &                                   # or: docker run -p 6379:6379 redis:7-alpine

cd backend
sh scripts/init_env.sh                           # once
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
sh scripts/download_models.sh                    # face models (skips files already present)
uvicorn app.main:app --reload --port 8000

cd ../frontend
npm install                                      # also copies the MediaPipe WASM into public/
sh scripts/download-face-model.sh                # optional: self-host the face-tracking model
npm run dev                                      # http://localhost:3000
```

The camera only works on `localhost` or HTTPS.

## Choosing the LLM adapter

The adapters (`backend/app/llm`) are the el-aiee ones, unchanged. `backend/.env` picks one:

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` | Google AI Studio |
| `gemini_vertex` | `backend/secrets/vertex-sa.json`, `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL` | GCP project billing |
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` | |
| `claude` | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` | |
| `deepseek` | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | Text only: use as `LLM_TEXT_PROVIDER` |

`LLM_TEXT_PROVIDER` (optional) runs the text-only step (address comparison) on a different adapter, e.g. `deepseek`.
`GET /health` shows which adapters are configured.

## What happens

1. **Login** – OAuth2 password flow (`POST /auth/token`) issues a JWT signed with `AUTH_SECRET_KEY`, stored in an httpOnly cookie.
   Optional "Continue with Google" when `GOOGLE_CLIENT_ID/SECRET` are set.
2. **Documents** – files are type-checked by their bytes, quality-checked (blur, glare, resolution) and encrypted at rest (`FILE_ENCRYPTION_KEY`).
   In the background the LLM reads the ID and the bill as structured JSON and flags tamper signs.
3. **Face check** – the backend issues a random head-turn order (stored in Redis, single use, 5-minute TTL).
   MediaPipe Face Landmarker runs in the browser to guide the user live and capture a frame when each pose is held.
4. **Server checks** – YuNet + SFace (OpenCV, CPU) re-check every frame:
   - one face per frame
   - the head really turned the requested way
   - the same person appears in every frame
   - the frames are not one replayed still image
   - the face matches the ID photo

   The LLM also reviews the frames for spoof signs (screens, prints, masks). Users get 3 attempts.
5. **Decision** – the rules engine produces pass, review or fail checks, then a decision:
   - any fail → `rejected`
   - any review → `manual_review`
   - otherwise → `approved`

   The checks:
   - NIC number format and its encoded birth date and gender against the printed details
   - passport MRZ check digits and expiry
   - bill is recent (≤ `BILL_MAX_AGE_DAYS`)
   - bill is in the ID holder's name
   - typed address matches the bill (fuzzy match plus LLM)
   - the same ID image was not used by another account
   - liveness
   - face match

Redis holds the short-lived state: liveness challenges, OAuth state, login rate limits, progress text and finalize locks.

## Tests

```bash
cd backend && pytest -q
```

The tests cover:

- NIC and MRZ rules
- name and address matching
- head-pose math
- the real face models, run on a public-domain photo
- the full API flow with a fake LLM adapter: approve, reject, retry, validation and access control

## MVP limits

- LLM tamper detection is a signal, not proof. Real verification needs an authoritative source (DRP NIC check, e-passport chip).
- Active liveness (random head turns) stops photos and simple replays, but it is not a certified presentation-attack detector.
- Tune the thresholds in `.env` with 50–100 real and fake samples before going live.
- There is no staff review screen yet. `manual_review` cases are visible in the database.
- Sending ID images to an external LLM API is a data transfer. Use API tiers with no training or retention, and get consent.
