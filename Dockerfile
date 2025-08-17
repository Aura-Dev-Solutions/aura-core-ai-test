FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar solo requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Descargar modelo de spaCy oficialmente
RUN python -m spacy download en_core_web_sm

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]