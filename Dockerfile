# Dockerfile for HF Spaces (sdk: docker). Must live at repo root.
FROM python:3.11-slim

# Make Python behave well in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps first so Docker can cache the pip layer when only
# source files change.
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install \
        "pydantic==2.12.5" \
        "fastapi==0.136.1" \
        "uvicorn[standard]==0.43.0" \
        "httpx==0.28.1"

# Now copy the rest of the source.
COPY market_env ./market_env
COPY client ./client

# HF Spaces routes external traffic to port 7860.
EXPOSE 7860

# Use the module form so import errors surface in the build logs.
CMD ["uvicorn", "market_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
