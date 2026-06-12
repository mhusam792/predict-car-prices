FROM python:3.13-slim

# environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (important for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

# copy dependency files only (for caching)
COPY pyproject.toml uv.lock ./

# install all dependencies once
RUN uv sync --frozen