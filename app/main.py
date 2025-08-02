from fastapi import FastAPI
import logging
from app.routers import documents

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir el ciclo de vida personalizado
async def lifespan(app: FastAPI):
    logger.info("Aplicación iniciada y lista para recibir solicitudes.")
    yield
    logger.info("Aplicación cerrada.")

# Inicialización de la aplicación FastAPI con ciclo de vida
app = FastAPI(lifespan=lifespan)

# Registrar el router para manejar las rutas relacionadas con documentos
app.include_router(documents.router)