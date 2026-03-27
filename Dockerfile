FROM python:3.11-slim

LABEL maintainer="arturwyroslak"
LABEL description="HuggingFace MCP Server — Hub management via Model Context Protocol"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY server.py .

# Non-root user for security
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser /app
USER mcpuser

# stdio transport — no port exposed
CMD ["python", "-u", "server.py"]
