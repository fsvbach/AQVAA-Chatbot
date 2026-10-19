#!/bin/sh
set -e

PROVIDER="${LLM_LLM_PROVIDER:-ollama}"

if [ "$PROVIDER" = "ollama" ]; then
    OLLAMA_URL="${LLM_OLLAMA_BASE_URL:-http://localhost:11434}"
    echo "Waiting for Ollama at $OLLAMA_URL..."
    until curl -sf "$OLLAMA_URL/api/tags" > /dev/null; do
        sleep 2
    done
    echo "Ollama is ready."

    for MODEL in "${LLM_LLM_MODEL:-llama3.2:3b}" "${LLM_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"; do
        echo "Pulling model: $MODEL (skipped if already present)"
        curl -sf -X POST "$OLLAMA_URL/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$MODEL\"}" | tr '\n' ' '
        echo ""
    done
else
    echo "Provider is '$PROVIDER' — skipping Ollama setup."
fi

exec python main.py
