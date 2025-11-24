# Tests y validación del sistema

Esta sección explica cómo validar que el sistema está funcionando correctamente.

No se cubren tests unitarios automatizados en detalle aquí, sino una validación funcional básica usando la API y archivos de prueba.

---

## 1. Prerrequisitos

- Proyecto en ejecución con Docker:
- API disponible en `http://localhost:8000`.
- Swagger disponible en `http://localhost:8000/docs`.

---

## 2. Archivo de prueba

Utiliza el archivo de prueba para poder validar el funcionamiento:

- [Archivo de prueba](docs/samples/test_file_core.pdf)  

Este archivo contiene texto muy sencillo sobre un tema en específico, puedes usarlo como referencia para hacer más pruebas.

---

## 3. Validar el flujo básico con Swagger

1. Abrir Swagger:

   ```text
   http://localhost:8000/docs
   ```

2. Probar `POST /documents`:
   - Click en `POST /documents`.
   - `Try it out`.
   - En `file`, seleccionar `docs/samples/test_document_ai_system.pdf`.
   - Ejecutar.
   - Verificar que la respuesta:
     - Tiene un `id`.
     - `status` sea `DONE`.
     - Aparece el campo `classification` con una categoría razonable.
     - Aparece `entities` con alguna entidad detectada.

3. Probar `GET /documents/{id}`:
   - Tomar el `id` que regresó el POST.
   - Ir a `GET /documents/{document_id}`.
   - `Try it out`.
   - Pegar el id y ejecutar.
   - Verificar que la información coincide con lo que se vio en el POST.

4. Probar `GET /search`:
   - Ir a `GET /search`.
   - `Try it out`.
   - En `q`, escribir una palabra que esté en el documento (por ejemplo `intelligence`, `AI`, etc.).
   - Ejecutar.
   - Verificar que en `results` aparece el documento que se subió.

---