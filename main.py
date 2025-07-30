from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.routes import get_routes
from app.db.database import init_db
import uvicorn
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_app():
    app = FastAPI(title="Document Analysis API")
    init_db()
    app.include_router(get_routes(), prefix="/api")

    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    return app


app = create_app()

if __name__ == "__main__":
    logger.info("Starting Document Analysis API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
