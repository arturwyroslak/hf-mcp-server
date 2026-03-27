FROM python:3.11-slim

LABEL maintainer="arturwyroslak"
LABEL description="HuggingFace MCP Server — Streamable HTTP transport on port 8000"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Create user and ALL required directories with correct ownership in one layer
RUN useradd -m -u 1000 mcpuser \
    && mkdir -p /home/mcpuser/.cache/huggingface/hub \
    && mkdir -p /tmp/hf_cache \
    && chown -R mcpuser:mcpuser /home/mcpuser \
    && chown -R mcpuser:mcpuser /tmp/hf_cache \
    && chown -R mcpuser:mcpuser /app

USER mcpuser

# Point HF cache to user-owned directory
ENV HF_HOME=/home/mcpuser/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/home/mcpuser/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/home/mcpuser/.cache/huggingface/hub

EXPOSE 8000

CMD ["python", "-u", "server.py"]
