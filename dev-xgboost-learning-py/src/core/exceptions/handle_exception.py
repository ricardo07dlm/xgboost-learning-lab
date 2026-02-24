from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions.service_error import ServiceError


def register_exception_handlers(app: FastAPI) -> FastAPI:

    # (1) ServiceError (tu error de capa Service)
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_key": str(exc),
                "details": exc.details
            }
        )

    # (2) RequestValidationError (FastAPI/Pydantic -> 422)
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error_key": "REQUEST_VALIDATION_ERROR",
                "details": exc.errors()
            }
        )

    # (3) HTTPException capa FASTAPI
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # exc.detail puede ser string o dict
        if isinstance(exc.detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error_key": exc.detail.get("error_key", "HTTP_ERROR"),
                    "details": exc.detail.get("details", exc.detail)
                }
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_key": "HTTP_ERROR",
                "details": exc.detail
            }
        )

    return app


