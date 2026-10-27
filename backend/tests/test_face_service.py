from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.face_result import FaceResult
from app.services import face_detection_service as service_module
from app.services.face_detection_service import (
    FaceDetectionService,
    InvalidImageError,
)


def _blank_png() -> bytes:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


class _FakeClassifier:
    def __init__(self, detections: list[tuple[int, int, int, int]]) -> None:
        self.detections = detections
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def empty(self) -> bool:
        return False

    def detectMultiScale(
        self, image: np.ndarray, **kwargs: object
    ) -> list[tuple[int, int, int, int]]:
        self.calls.append((image, kwargs))
        return self.detections


def _service_with_detections(
    monkeypatch: pytest.MonkeyPatch,
    detections: list[tuple[int, int, int, int]],
) -> tuple[FaceDetectionService, _FakeClassifier]:
    classifier = _FakeClassifier(detections)
    monkeypatch.setattr(
        service_module.cv2,
        "CascadeClassifier",
        lambda _: classifier,
    )
    return FaceDetectionService(), classifier


def test_detects_a_face_and_uses_configured_cascade_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, classifier = _service_with_detections(
        monkeypatch, [(10, 12, 40, 40)]
    )

    result = service.detect(_blank_png())

    assert result == FaceResult(
        is_face=True,
        faces_count=1,
        message="Se detectó al menos un rostro.",
        face_boxes=((10, 12, 40, 40),),
    )
    assert len(classifier.calls) == 1
    grayscale, parameters = classifier.calls[0]
    assert grayscale.ndim == 2
    assert parameters == {
        "scaleFactor": 1.1,
        "minNeighbors": 5,
        "minSize": (40, 40),
    }


def test_counts_multiple_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ = _service_with_detections(
        monkeypatch,
        [(1, 2, 40, 40), (45, 2, 40, 40)],
    )

    result = service.detect(_blank_png())

    assert result.is_face is True
    assert result.faces_count == 2


def test_annotates_detected_forehead_and_returns_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, classifier = _service_with_detections(
        monkeypatch, [(20, 10, 60, 80)]
    )

    result, annotated_bytes = service.annotate(_blank_png())

    assert result.is_face is True
    assert result.face_boxes == ((20, 10, 60, 80),)
    assert annotated_bytes is not None
    decoded = cv2.imdecode(
        np.frombuffer(annotated_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    marker_pixels = decoded[20:37, 40:60]
    assert np.any(
        (marker_pixels[:, :, 2] > 200)
        & (np.abs(marker_pixels[:, :, 0].astype(int) - 56) < 20)
        & (np.abs(marker_pixels[:, :, 1].astype(int) - 48) < 20)
    )
    label_pixels = decoded[10:20, 30:70]
    assert np.any(
        (label_pixels[:, :, 2] > 200)
        & (np.abs(label_pixels[:, :, 0].astype(int) - 56) < 20)
        & (np.abs(label_pixels[:, :, 1].astype(int) - 48) < 20)
    )
    assert len(classifier.calls) == 1


def test_does_not_create_an_image_when_no_face_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = _service_with_detections(monkeypatch, [])

    result, annotated_bytes = service.annotate(_blank_png())

    assert result.is_face is False
    assert annotated_bytes is None


def test_downscales_large_images_and_restores_face_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, classifier = _service_with_detections(
        monkeypatch, [(400, 100, 100, 100)]
    )
    image = np.zeros((1000, 1200, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    result = service.detect(encoded.tobytes())

    assert classifier.calls[0][0].shape == (667, 800)
    assert result.face_boxes == ((600, 150, 150, 150),)


def test_real_classifier_returns_no_face_for_blank_image() -> None:
    result = FaceDetectionService().detect(_blank_png())

    assert result.is_face is False
    assert result.faces_count == 0
    assert result.message == "No se detectaron rostros."


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_rejects_empty_or_undecodable_bytes(image_bytes: bytes) -> None:
    service = FaceDetectionService()

    with pytest.raises(InvalidImageError):
        service.detect(image_bytes)


def test_fails_explicitly_when_classifier_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyClassifier:
        def empty(self) -> bool:
            return True

    monkeypatch.setattr(
        service_module.cv2,
        "CascadeClassifier",
        lambda _: EmptyClassifier(),
    )

    with pytest.raises(RuntimeError, match="could not load"):
        FaceDetectionService()


@pytest.mark.parametrize(
    "sample_name",
    ["rostro.jpg", "varios-rostros.jpg", "sin-rostros.jpg"],
)
def test_optional_real_photo_samples(sample_name: str) -> None:
    sample_path = (
        Path(__file__).parents[1] / "assets" / "rostros" / sample_name
    )
    if not sample_path.is_file():
        pytest.skip(
            f"Falta la foto autorizada {sample_path.name}; "
            "la precisión real no se valida sin esta muestra."
        )

    result = FaceDetectionService().detect(sample_path.read_bytes())
    if sample_name == "rostro.jpg":
        assert result.is_face is True
    elif sample_name == "varios-rostros.jpg":
        assert result.faces_count >= 2
    else:
        assert result.is_face is False
