FROM python:3.11-slim

WORKDIR /app

# Install curl + uvloop + httptools build deps in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN useradd -m -u 1000 mcpuser \
    && chown -R mcpuser:mcpuser /app

USER mcpuser

ENV HF_HOME=/tmp/hf
ENV HUGGINGFACE_HUB_CACHE=/tmp/hf
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-u", "server.py"]
