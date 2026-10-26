from collections.abc import Iterable

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.config import MAX_UPLOAD_SIZE_BYTES
from app.models.face_result import FaceResult
from app.services.face_detection_service import (
    FaceDetectionService,
    InvalidImageError,
)
from app.views.face_view import face_result_to_response

router = APIRouter()
face_detection_service = FaceDetectionService()
_READ_CHUNK_SIZE = 64 * 1024
_SUPPORTED_MEDIA_TYPES = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),
}


def _raise_http_error(status_code: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


async def _read_limited(file: UploadFile, max_size: int) -> bytes:
    content = bytearray()
    while len(content) <= max_size:
        chunk_size = min(_READ_CHUNK_SIZE, max_size - len(content) + 1)
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_size:
            _raise_http_error(
                413,
                "FILE_TOO_LARGE",
                "El archivo supera el tamaño máximo permitido.",
            )
    return bytes(content)


def _has_supported_signature(content: bytes, media_type: str) -> bool:
    signatures = _SUPPORTED_MEDIA_TYPES[media_type]
    if not any(content.startswith(signature) for signature in signatures):
        return False
    if media_type == "image/webp":
        return len(content) >= 12 and content[8:12] == b"WEBP"
    return True


def _uploaded_files(form: FormData) -> Iterable[StarletteUploadFile]:
    return (
        value
        for _, value in form.multi_items()
        if isinstance(value, StarletteUploadFile)
    )


async def _read_image_upload(request: Request, file: UploadFile) -> bytes:
    uploaded_files = list(_uploaded_files(await request.form()))
    if len(uploaded_files) != 1 or uploaded_files[0] is not file:
        _raise_http_error(
            400,
            "INVALID_FILE_COUNT",
            "La solicitud debe contener exactamente un archivo.",
        )

    media_type = (file.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if media_type not in _SUPPORTED_MEDIA_TYPES:
        _raise_http_error(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "El formato debe ser JPEG, PNG o WebP.",
        )

    content = await _read_limited(file, MAX_UPLOAD_SIZE_BYTES)
    if not content:
        _raise_http_error(400, "EMPTY_FILE", "El archivo está vacío.")
    if not _has_supported_signature(content, media_type):
        _raise_http_error(
            400,
            "INVALID_IMAGE",
            "El archivo no contiene una imagen válida del tipo indicado.",
        )
    return content


@router.post("/api/v1/face/validate")
async def validate_face(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, object]:
    content = await _read_image_upload(request, file)
    try:
        result: FaceResult = await run_in_threadpool(
            face_detection_service.detect, content
        )
    except InvalidImageError:
        _raise_http_error(
            400,
            "INVALID_IMAGE",
            "El archivo no contiene una imagen válida.",
        )
    return face_result_to_response(result)


@router.post(
    "/api/v1/face/annotate",
    response_class=Response,
    responses={
        200: {
            "description": "Imagen PNG anotada con la frente detectada.",
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        404: {"description": "No se detectó ningún rostro."},
    },
)
async def annotate_face(
    request: Request,
    file: UploadFile = File(...),
) -> Response:
    content = await _read_image_upload(request, file)
    try:
        result, annotated_image = await run_in_threadpool(
            face_detection_service.annotate, content
        )
    except InvalidImageError:
        _raise_http_error(
            400,
            "INVALID_IMAGE",
            "El archivo no contiene una imagen válida.",
        )

    if not result.is_face or annotated_image is None:
        _raise_http_error(
            404,
            "FACE_NOT_DETECTED",
            "No se detectó un rostro; no se generó ninguna imagen.",
        )
    return Response(content=annotated_image, media_type="image/png")
