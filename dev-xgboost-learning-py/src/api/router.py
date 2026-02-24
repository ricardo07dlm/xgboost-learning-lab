from fastapi import FastAPI

from core.exceptions.handle_exception import register_exception_handlers
from src.api.controller import xgboost_controller
from core.env.settings_config import load_config

# Crear instancia de FastAPI
app = FastAPI()

register_exception_handlers(app)


@app.on_event("startup")
def startup():
    load_config()


# Incluir router de signUp
app.include_router(xgboost_controller.router, prefix="/xgboost", tags=["xgboost"])
