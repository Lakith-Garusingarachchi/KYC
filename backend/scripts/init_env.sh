#!/usr/bin/env sh
# Creates backend/.env from .env.example with a fresh AUTH_SECRET_KEY and FILE_ENCRYPTION_KEY.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  echo "backend/.env already exists — not overwriting"
  exit 0
fi
cp .env.example .env
AUTH=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))" 2>/dev/null \
  || openssl rand -base64 48 | tr -d '\n/+=')
FERNET=$(python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())" 2>/dev/null \
  || openssl rand -base64 32 | tr '+/' '-_')
sed -i.bak "s|^AUTH_SECRET_KEY=.*|AUTH_SECRET_KEY=$AUTH|; s|^FILE_ENCRYPTION_KEY=.*|FILE_ENCRYPTION_KEY=$FERNET|" .env
rm -f .env.bak
echo "backend/.env created with new secrets."
echo "Next: set LLM_PROVIDER and that provider's API key (e.g. GEMINI_API_KEY) in backend/.env"
