FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY evalscope/web/package*.json ./
RUN npm ci --registry=https://registry.npmmirror.com
COPY evalscope/web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EVALSCOPE_OUTPUT_DIR=/app/outputs HF_ENDPOINT=https://hf-mirror.com \
    REGISTRATION_MODE=admin_only EVALSCOPE_ENABLE_LEGACY_DOCKER_CLEANUP=0
RUN python -c "from pathlib import Path; p=Path('/etc/apt/sources.list.d/debian.sources'); p.write_text(p.read_text().replace('http://deb.debian.org','https://mirrors.aliyun.com'))" \
    && apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates git curl ffmpeg libsndfile1 libgomp1 build-essential docker.io tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu -r /opt/requirements.lock \
    && pip check
COPY . /app/
COPY --from=frontend /web/dist /app/evalscope/web/dist
RUN pip install --no-deps --no-build-isolation -e . \
    && mkdir -p /app/outputs
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health',timeout=4)"
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/entrypoint.py"]
