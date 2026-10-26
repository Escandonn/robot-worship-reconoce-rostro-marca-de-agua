from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.controllers.face_controller import router as face_router
from app.middleware import RequestBodyLimitMiddleware

app = FastAPI(title="WORSHIP Face Detection API", version="1.0.0")
app.add_middleware(RequestBodyLimitMiddleware)
app.include_router(face_router)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    missing_file = any(
        item.get("type") == "missing" and "file" in item.get("loc", ())
        for item in error.errors()
    )
    code = "MISSING_FILE" if missing_file else "INVALID_REQUEST"
    message = (
        "La solicitud no contiene el archivo requerido."
        if missing_file
        else "La solicitud no es válida."
    )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {"code": code, "message": message},
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    if isinstance(error.detail, dict):
        code = error.detail.get("code", "REQUEST_ERROR")
        message = error.detail.get("message", "La solicitud no pudo procesarse.")
    else:
        code = "REQUEST_ERROR"
        message = "La solicitud no pudo procesarse."
    return JSONResponse(
        status_code=error.status_code,
        content={
            "success": False,
            "error": {"code": code, "message": message},
        },
        headers=error.headers,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
