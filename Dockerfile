FROM python:3.12-slim

# Evitar preguntas interactivas
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear y usar directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN pip install python-multipart


# Copiar el resto del código
COPY . .

# Comando para correr el script main.py directamente
CMD ["python", "main.py"]
