# ---- dependencies ----
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
COPY scripts ./scripts
RUN npm ci --no-audit --no-fund

# ---- build ----
FROM node:22-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
RUN apk add --no-cache curl
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# MediaPipe WASM runtime + face landmark model are served from /mediapipe.
# If the model download fails, the browser falls back to Google's hosted copy.
RUN node scripts/copy-mediapipe.mjs \
    && (sh scripts/download-face-model.sh || echo "WARN: face model download failed; using hosted fallback")
RUN npm run build \
    && chmod -R a+rX public .next/standalone .next/static

# ---- runtime ----
FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
RUN addgroup -S app && adduser -S app -G app
COPY --from=build --chown=app:app /app/public ./public
COPY --from=build --chown=app:app /app/.next/standalone ./
COPY --from=build --chown=app:app /app/.next/static ./.next/static
USER app
EXPOSE 3000
CMD ["node", "server.js"]
