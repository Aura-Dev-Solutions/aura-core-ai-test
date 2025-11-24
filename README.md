# AURA Core AI – Document Analysis System

AURA Core AI es un sistema para analizar documentos con una estructura muy sencilla de título, contenido y conclusiones usando IA.  
Permite:

- Subir documentos por **API**.
- Guardarlos y extraer el contenido.
- Clasificarlos por tema.
- Detectar entidades dentro del contenido.
- Hacer búsquedas sobre los documentos procesados.

---

## Requisitos

- Git
- Docker Desktop 4.x.x o superior

---

## Step by Step

- Clonar el repositorio.
- Copiar las variables del `.env.example`:

```bash
cp .env.example .env
```

Nota: Los valores para que el proyecto pueda estar en funcionamiento viven en el archivo .env (este proceso sólo es por si se quieren adaptar los valores de forma personalizada).

---

## Cómo ejecutar el proyecto con Docker

Desde la raíz del repositorio:

```bash
docker compose build
docker compose up
```

Una vez que los servicios estén en ejecución:

- API: `http://localhost:8000`
- Swagger (documentación interactiva): `http://localhost:8000/docs`

---

## Estructura del proyecto

El código está organizado por responsabilidad, para que cada parte tenga un propósito claro:

```text
app/
  main.py
  api/
  domain/
  models/
  files/
  ml/
  config.py
tools/
  folder_watcher.py
docs/
  README.md
  ml/
  api/
  test/
```

- `app/api`: todo lo relacionado con la API (endpoints HTTP).
- `app/domain`: lógica de negocio (ingestar, procesar y buscar documentos).
- `app/models`: modelos de datos, entidades, esquemas y acceso a base de datos.
- `app/files`: manejo de archivos físicos y extracción de texto.
- `app/ml`: modelos y lógica de Machine Learning.
- `tools/folder_watcher.py`: proceso considerado para tomar archivos de una carpeta en específico (Nice to have - No implementado).
- `docs/`: documentación más detallada del proyecto.

Esta estructura ayuda a que sea más fácil cambiar o mejorar una parte (por ejemplo, el modelo de ML o la base de datos) sin afectar el resto del sistema.

---

## Documentación detallada

La explicación completa del sistema está dividida en varias secciones dentro de `docs/`:

- [Machine Learning](docs/001_ML.md)  
  Detalles de los modelos de IA: embeddings, clasificación y NER.

- [API](docs/002_API.md)  
  Endpoints disponibles, uso de Swagger y ejemplos de uso.

- [Tests y validación](docs/test/README.md)  
  Cómo probar el sistema, flujo de validación y archivos de prueba.
