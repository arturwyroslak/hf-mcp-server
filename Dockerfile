FROM python:3.11-slim

LABEL maintainer="arturwyroslak"
LABEL description="HuggingFace MCP Server — Streamable HTTP transport on port 8000"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN useradd -m -u 1000 mcpuser && chown -R mcpuser /app
USER mcpuser

EXPOSE 8000

CMD ["python", "-u", "server.py"]
