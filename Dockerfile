FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN python -m pip install --upgrade pip &&     pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
