import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.controllers import face_controller
from app.main import app
from app.models.face_result import FaceResult

client = TestClient(app)


def _blank_png() -> bytes:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


class _StubDetectionService:
    def __init__(self, result: FaceResult) -> None:
        self.result = result

    def detect(self, image_bytes: bytes) -> FaceResult:
        assert image_bytes
        return self.result


class _StubAnnotationService(_StubDetectionService):
    def annotate(self, image_bytes: bytes) -> tuple[FaceResult, bytes | None]:
        assert image_bytes
        if not self.result.is_face:
            return self.result, None
        return self.result, b"annotated-png"


def _set_result(monkeypatch, *, is_face: bool, faces_count: int) -> None:
    message = (
        "Se detectó al menos un rostro."
        if is_face
        else "No se detectaron rostros."
    )
    monkeypatch.setattr(
        face_controller,
        "face_detection_service",
        _StubDetectionService(
            FaceResult(is_face=is_face, faces_count=faces_count, message=message)
        ),
    )


def _set_annotation_result(
    monkeypatch, *, is_face: bool, faces_count: int
) -> None:
    message = (
        "Se detectó al menos un rostro."
        if is_face
        else "No se detectaron rostros."
    )
    monkeypatch.setattr(
        face_controller,
        "face_detection_service",
        _StubAnnotationService(
            FaceResult(is_face=is_face, faces_count=faces_count, message=message)
        ),
    )


def _assert_error_contract(response, status_code: int) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert body["success"] is False
    assert set(body["error"]) == {"code", "message"}


def test_endpoint_returns_the_face_response_contract(
    monkeypatch,
) -> None:
    _set_result(monkeypatch, is_face=True, faces_count=1)

    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("face.png", _blank_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "is_face": True,
        "faces_count": 1,
        "message": "Se detectó al menos un rostro.",
    }


def test_endpoint_returns_success_when_no_face_is_detected(
    monkeypatch,
) -> None:
    _set_result(monkeypatch, is_face=False, faces_count=0)

    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("blank.png", _blank_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["is_face"] is False


def test_annotate_endpoint_returns_png_only_when_face_is_detected(
    monkeypatch,
) -> None:
    _set_annotation_result(monkeypatch, is_face=True, faces_count=1)

    response = client.post(
        "/api/v1/face/annotate",
        files={"file": ("face.png", _blank_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"annotated-png"


def test_annotate_endpoint_returns_error_without_an_image_when_no_face(
    monkeypatch,
) -> None:
    _set_annotation_result(monkeypatch, is_face=False, faces_count=0)

    response = client.post(
        "/api/v1/face/annotate",
        files={"file": ("blank.png", _blank_png(), "image/png")},
    )

    _assert_error_contract(response, 404)
    assert response.json()["error"]["code"] == "FACE_NOT_DETECTED"


def test_missing_file_uses_the_documented_error_contract() -> None:
    response = client.post("/api/v1/face/validate")

    _assert_error_contract(response, 422)
    assert response.json()["error"]["code"] == "MISSING_FILE"


def test_empty_file_returns_a_client_error() -> None:
    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("empty.png", b"", "image/png")},
    )

    _assert_error_contract(response, 400)
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_corrupt_image_is_not_reported_as_a_negative_detection() -> None:
    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("broken.png", b"\x89PNG\r\n\x1a\nbroken", "image/png")},
    )

    _assert_error_contract(response, 400)
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_unsupported_media_type_returns_415() -> None:
    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("text.txt", b"hello", "text/plain")},
    )

    _assert_error_contract(response, 415)
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_larger_than_configured_limit_returns_413(
    monkeypatch,
) -> None:
    monkeypatch.setattr(face_controller, "MAX_UPLOAD_SIZE_BYTES", 4)

    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("large.png", _blank_png(), "image/png")},
    )

    _assert_error_contract(response, 413)
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_request_body_limit_rejects_oversized_stream() -> None:
    oversized_file = b"x" * (5 * 1024 * 1024 + 64 * 1024)
    response = client.post(
        "/api/v1/face/validate",
        files={"file": ("large.png", oversized_file, "image/png")},
    )

    _assert_error_contract(response, 413)
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_annotate_endpoint_applies_request_body_limit() -> None:
    oversized_file = b"x" * (5 * 1024 * 1024 + 64 * 1024)
    response = client.post(
        "/api/v1/face/annotate",
        files={"file": ("large.png", oversized_file, "image/png")},
    )

    _assert_error_contract(response, 413)
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_rejects_multiple_uploaded_files() -> None:
    image = _blank_png()
    response = client.post(
        "/api/v1/face/validate",
        files=[
            ("file", ("one.png", image, "image/png")),
            ("another", ("two.png", image, "image/png")),
        ],
    )

    _assert_error_contract(response, 400)
    assert response.json()["error"]["code"] == "INVALID_FILE_COUNT"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
