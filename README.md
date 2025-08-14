# Aura Core AI

Aura Core AI es una plataforma modular para procesamiento inteligente de documentos, extracción de texto, generación de embeddings y búsqueda semántica. Incluye servicios de clasificación básica y una API REST lista para producción.

## Funcionalidades principales

- **Ingesta y extracción de texto** desde archivos PDF, DOCX y JSON.
- **Procesamiento paralelo** para mayor velocidad.
- **Generación de embeddings** usando modelos de Sentence-Transformers.
- **Búsqueda semántica** eficiente con FAISS.
- **Clasificación básica** (TF-IDF + Regresión Logística) y opción zero-shot.
- **API REST** con FastAPI, diseño limpio y modular.
- **Soporte multilenguaje** (español e inglés) para clasificacion de documento.
- **Contenedores Docker** y `docker-compose` para desarrollo local.
- **Pruebas automáticas** con Pytest.
- **Configuración centralizada** con `pyproject.toml`.

## Quickstart

```bash
# 1) Construir y ejecutar con Docker (recomendado)
docker compose up --build

# 2) O ejecutar localmente (Python 3.10+ recomendado)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Una vez corriendo, abre [http://localhost:8000/docs](http://localhost:8000/docs) para explorar la API.

---

## Ejemplos de uso de la API

### 1. Ingestar un documento

**Bash (cURL):**
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@/ruta/a/tu/archivo.pdf"
```

**Postman:**
- Método: `POST`
- URL: `http://localhost:8000/api/ingest`
- Body: Form-data, clave `file`, valor: selecciona tu archivo.

---

### 2. Buscar texto semánticamente

**Bash (cURL):**
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Buscar información relevante", "top_k": 5}'
```

**Postman:**
- Método: `POST`
- URL: `http://localhost:8000/api/search`
- Body: raw, JSON:
  ```json
  {
    "query": "Buscar información relevante",
    "top_k": 5
  }
  ```

---

### 3. Clasificar un texto

**Bash (cURL):**
```bash
curl -X POST "http://localhost:8000/api/classify" \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "Id del documento."}'
```

**Postman:**
- Método: `POST`
- URL: `http://localhost:8000/api/classify`
- Body: raw, JSON:
  ```json
  {
    "doc_id": "Id del documento."
  }
  ```

---

## Estructura del proyecto

```
app/
  api/
    routes.py            # Endpoints FastAPI
  services/
    extract.py           # Extracción de texto
    embeddings.py        # Generación de embeddings
    index.py             # Índice FAISS
    search.py            # Búsqueda semántica
    classify.py          # Clasificador
  storage/
    db.py                # Base de datos SQLite
  config.py              # Configuración
  models.py              # Modelos Pydantic
  main.py                # App FastAPI
tests/
  test_extract.py
  test_embeddings.py
  test_search.py
Dockerfile
docker-compose.yml
requirements.txt
pyproject.toml           # Configuración de herramientas
```

---