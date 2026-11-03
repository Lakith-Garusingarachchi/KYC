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
# creates (or cleans up) backend/.env: keeps existing values, generates the auth secret
# and file-encryption key, and takes overrides from the environment
LLM_PROVIDER=claude ANTHROPIC_API_KEY=sk-ant-... sh backend/scripts/init_env.sh
docker compose up --build
```

After changing `backend/.env`, restart the backend: `docker compose up -d --force-recreate backend`.

- App: http://localhost:3000
- API docs: http://localhost:8000/docs

### Deploy on a server with Cloudflare Tunnel (no nginx, no open ports)

1. In the Cloudflare dashboard: **Zero Trust → Networks → Tunnels → Create a tunnel** (type *Cloudflared*).
   Copy the tunnel token. Under **Public hostname** add e.g. `kyc.example.com` → service `HTTP` → `frontend:3000`.
2. On the server:
   ```bash
   git clone <repo> KYC && cd KYC
   sh backend/scripts/init_env.sh                 # new secrets for this server; then add the LLM key
   echo "CLOUDFLARE_TUNNEL_TOKEN=<token>" > .env  # read by docker compose
   docker compose --profile tunnel up -d --build
   ```
3. In `backend/.env` set `FRONTEND_URL=https://kyc.example.com`, `CORS_ORIGINS=https://kyc.example.com`,
   `AUTH_COOKIE_SECURE=true` (and `GOOGLE_REDIRECT_URI=https://kyc.example.com/api/auth/google/callback` if you use Google login),
   then `docker compose up -d --force-recreate backend`.

Ports 3000/8000 are bound to 127.0.0.1, so the app is only reachable through the tunnel.
Back up the `kyc_backend-data` volume **and** `FILE_ENCRYPTION_KEY` together — the files can't be read without the key.

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
5. **Decision** – every check comes out pass, review or fail, and adds up to a **verification score** (0–100%).
   Each check has a weight (face match 20, address match 10, liveness 10…); a pass counts fully, a review counts half,
   a fail counts zero. The result is **Passed** (`approved`) or **Manual review required** (`manual_review`):
   - score below `KYC_PASS_SCORE` (default 40) → manual review
   - any check in `KYC_REQUIRED_CHECKS` not passed (default: face match, head turns, same person, address match) → manual review
   - a hard failure (not an ID, invalid NIC number, expired document, not a bill, bill too old, replayed still image) → manual review
   - otherwise → passed

   Minor document notes (wear, glare, normal meter-reader handwriting) don't count against the score; only major tamper signs do.
   The result page shows the score, the reasons for review and every check, and can be saved as a PDF.

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
- Tune the thresholds with 50–100 real and fake samples before going live. They default in `backend/app/config.py` and can be overridden in `.env` (e.g. `FACE_MATCH_THRESHOLD`, `LIVENESS_YAW_DELTA`, `BILL_MAX_AGE_DAYS`).
- There is no staff review screen yet. `manual_review` cases are visible in the database.
- Sending ID images to an external LLM API is a data transfer. Use API tiers with no training or retention, and get consent.
