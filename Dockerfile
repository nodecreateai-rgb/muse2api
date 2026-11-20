FROM python:3.12-slim

ARG APT_MIRROR=
ARG PIP_INDEX_URL=

RUN if [ -n "$APT_MIRROR" ]; then \
        [ -f /etc/apt/sources.list.d/debian.sources ] && sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources || true; \
        [ -f /etc/apt/sources.list ] && sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list || true; \
    fi \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8090 \
    DATABASE_URL=sqlite:///./data/muse-proxy.db \
    ENGINE_BACKEND=protocol

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} -r /app/requirements.txt

COPY app /app/app
COPY static /app/static
COPY .env.example /app/.env.example
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/data /app/data/media /app/data/downloads /app/data/profiles

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8090}/healthz" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
