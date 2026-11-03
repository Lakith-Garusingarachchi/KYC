FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Face models: reuse the ones in backend/models if present, otherwise download them.
COPY scripts/ scripts/
COPY models/ models/
RUN sh scripts/download_models.sh

COPY app/ app/

# Host files can arrive with owner-only permissions (umask 077); make the code
# readable for the non-root runtime user.
RUN useradd --create-home --uid 1000 kyc \
    && chmod -R a+rX /app \
    && mkdir -p /app/data/uploads /app/secrets \
    && chown -R kyc:kyc /app/data
USER kyc

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
