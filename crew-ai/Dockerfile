FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root
COPY . .
RUN poetry install
COPY <<EOF /entrypoint.sh
#!/bin/sh

if test -f /run/secrets/openai-api-key; then
    export OPENAI_API_KEY=$(cat /run/secrets/openai-api-key)
fi

if test -n "\${OPENAI_API_KEY}"; then
    echo "Using OpenAI with \${OPENAI_MODEL_NAME}"
else
    echo "Using Docker Model Runner with \${MODEL_RUNNER_MODEL}"
    # Use hosted_vllm prefix which properly handles messages for local models
    export OPENAI_BASE_URL=\${MODEL_RUNNER_URL}
    export OPENAI_MODEL_NAME=hosted_vllm/\${MODEL_RUNNER_MODEL}
    # Set drop_params to handle consecutive message issues
    export LITELLM_DROP_PARAMS=True
    export OPENAI_API_KEY=cannot_be_empty
fi
exec poetry run marketing_posts
EOF
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
