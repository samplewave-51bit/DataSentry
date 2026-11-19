# Multi-stage Dockerfile for DataSentry Enterprise Service
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[api,dev]

# Copy application source code and configuration policies
COPY src/ /app/src/
COPY policies/ /app/policies/
COPY config/ /app/config/

# Create non-root user for enterprise container hardening
RUN groupadd -r datasentry && useradd -r -g datasentry -m datasentry && \
    mkdir -p /app/logs && chown -R datasentry:datasentry /app
USER datasentry

# Expose FastAPI REST service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch production ASGI server
CMD ["uvicorn", "datasentry.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
