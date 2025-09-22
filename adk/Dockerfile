FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1

# Toolchain for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ make pkg-config git curl \
  && rm -rf /var/lib/apt/lists/*

# Rust for crates-backed Python packages
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# uv
RUN pip install uv

WORKDIR /app

# Copy lock/metadata first to leverage layer caching
COPY pyproject.toml uv.lock ./

# Install deps (will compile missing wheels on 3.13)
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    uv pip install --system . && \
    rm -fr build dist *.egg-info

# Copy application code
COPY agents/ ./agents/
RUN python -m compileall -q .

# Write entrypoint (COPY <<EOF is invalid in Dockerfile)
RUN <<'EOF' bash -euo pipefail
cat >/entrypoint.sh <<'SH'
#!/bin/sh
set -e

if [ -f /run/secrets/openai-api-key ]; then
  export OPENAI_API_KEY="$(cat /run/secrets/openai-api-key)"
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "Using OpenAI with ${OPENAI_MODEL_NAME:-<unset>}"
else
  echo "Using Docker Model Runner with ${MODEL_RUNNER_MODEL:-<unset>}"
  export OPENAI_BASE_URL="${MODEL_RUNNER_URL}"
  export OPENAI_MODEL_NAME="openai/${MODEL_RUNNER_MODEL}"
  export OPENAI_API_KEY="cannot_be_empty"
fi

exec adk web --host 0.0.0.0 --port 8080 --log_level DEBUG
SH
chmod +x /entrypoint.sh
EOF

# Non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

ENTRYPOINT ["/entrypoint.sh"]

