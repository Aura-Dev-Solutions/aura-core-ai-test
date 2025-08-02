import asyncio
from fastapi import APIRouter, UploadFile, File
from concurrent.futures import ThreadPoolExecutor
from app.models import embeddings, classifier, ner_model
from app.models.extract_text import extract_text
from app.models.document_store import DocumentStore
from fastapi import HTTPException



router = APIRouter()
document_store = DocumentStore()

def process_document(file_content: bytes, filename: str):
    try:
        text = extract_text(file_content, filename)
        embedding = embeddings.generate_embeddings(text)
        classification = classifier.classify(text)
        entities = ner_model.extract_entities(text)

        document_store.add_document(embedding, filename, {"classification": classification})

        return {
            "filename": filename,
            "classification": classification,
            "entities": entities,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding) if isinstance(embedding, (list, tuple)) else embedding
        }
    except Exception as e:
        print(f"Error procesando {filename}: {str(e)}")
        return {
            "filename": filename,
            "error": str(e)
        }

@router.post("/process-multiple")
async def process_multiple_documents(files: list[UploadFile] = File(...)):
    try:
        print("Intentando leer archivos...")
        contents = await asyncio.gather(*(file.read() for file in files))
        print("Archivos leídos con éxito:", len(contents))
        filenames = [file.filename for file in files]
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = [executor.submit(process_document, content, fname) 
                              for content, fname in zip(contents, filenames)]
            for i, future in enumerate(future_results):
                try:
                    result = future.result()
                    print(f"Resultado {i} tipo: {type(result).__name__}")
                    if isinstance(result, dict):
                        print("Resultado para", result.get("filename", "desconocido"))
                    else:
                        print(f"Resultado inesperado {i}: {result}")
                    results.append(result)
                except Exception as e:
                    print(f"Error obteniendo resultado {i}: {str(e)}")
                    results.append({"filename": filenames[i], "error": str(e)})

        print("Procesamiento completado, devolviendo resultados...")
        return results
    except Exception as e:
        print("Error en el procesamiento:", str(e))
        raise


@router.post("/search")
async def search_documents(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Consulta vacía")
    
    # Generar embedding para la consulta
    query_embedding = embeddings.generate_embeddings(query)
    
    # Buscar los documentos más similares
    results = document_store.search(query_embedding, k=5)
    return {"query": query, "results": results}