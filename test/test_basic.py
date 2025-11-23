import pytest
import httpx
import time

BASE_URL = "http://0.0.0.0:8000"


def wait_for_api(timeout: int = 30):
    """Espera hasta que la API esté lista (útil en CI/CD)"""
    print(f"Esperando a que la API esté lista en {BASE_URL}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if response.status_code == 200:
                print("API lista!")
                return True
        except httpx.ConnectError:
            time.sleep(1)
    
    raise TimeoutError(f"La API no responde después de {timeout} segundos")


def test_api_is_up():
    """Test básico: la API está levantada y responde 200"""
    response = httpx.get(f"{BASE_URL}/docs", timeout=10.0)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Swagger UI" in response.text or "Redoc" in response.text


def test_health_endpoint():
    """Test explícito del endpoint /health (recomendado)"""

    wait_for_api()
    
    response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "aura-api"


def test_openapi_json():
    """Verifica que el schema OpenAPI está disponible"""
    response = httpx.get(f"{BASE_URL}/openapi.json", timeout=5.0)
    assert response.status_code == 200
    openapi = response.json()
    assert openapi["info"]["title"] == "Aura RAG API" 
    assert "paths" in openapi