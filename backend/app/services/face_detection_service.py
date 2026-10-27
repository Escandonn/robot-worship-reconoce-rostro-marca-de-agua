import cv2
import numpy as np

from app.models.face_result import FaceResult

_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_MAX_DETECTION_DIMENSION = 800


class InvalidImageError(ValueError):
    """Raised when the supplied bytes cannot be decoded as an image."""


class FaceDetectionService:
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (40, 40),
    ) -> None:
        cascade_path = cv2.data.haarcascades + _CASCADE_FILENAME
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError(
                f"OpenCV could not load the Haar Cascade classifier: {cascade_path}"
            )
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, image_bytes: bytes) -> FaceResult:
        image = self._decode_image(image_bytes)
        return self._detect_image(image)

    def annotate(self, image_bytes: bytes) -> tuple[FaceResult, bytes | None]:
        image = self._decode_image(image_bytes)
        result = self._detect_image(image)
        if not result.is_face:
            return result, None

        for x, y, width, height in result.face_boxes:
            self._draw_forehead_marker(image, x, y, width, height)

        encoded, output = cv2.imencode(".png", image)
        if not encoded:
            raise RuntimeError("OpenCV could not encode the annotated image.")
        return result, output.tobytes()

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("Image content is empty.")

        try:
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except cv2.error as error:
            raise InvalidImageError("Image content could not be decoded.") from error
        if image is None or image.size == 0:
            raise InvalidImageError("Image content could not be decoded.")
        return image

    def _detect_image(self, image: np.ndarray) -> FaceResult:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = grayscale.shape
        scale = min(1.0, _MAX_DETECTION_DIMENSION / max(width, height))
        if scale < 1.0:
            grayscale = cv2.resize(
                grayscale,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        faces = self._classifier.detectMultiScale(
            grayscale,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        inverse_scale = 1.0 / scale
        face_boxes = tuple(
            (
                round(x * inverse_scale),
                round(y * inverse_scale),
                round(face_width * inverse_scale),
                round(face_height * inverse_scale),
            )
            for x, y, face_width, face_height in faces
        )
        faces_count = len(face_boxes)
        is_face = faces_count > 0
        message = (
            "Se detectó al menos un rostro."
            if is_face
            else "No se detectaron rostros."
        )
        return FaceResult(
            is_face=is_face,
            faces_count=faces_count,
            message=message,
            face_boxes=face_boxes,
        )

    @staticmethod
    def _draw_forehead_marker(
        image: np.ndarray,
        face_x: int,
        face_y: int,
        face_width: int,
        face_height: int,
    ) -> None:
        center = (
            face_x + face_width // 2,
            face_y + round(face_height * 0.22),
        )
        radius = max(4, round(min(face_width * 0.13, face_height * 0.09)))
        color = (56, 48, 255)
        thickness = max(1, round(face_width / 120))
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)

        triangle_radius = radius * 0.75
        triangle = np.array(
            [
                (
                    center[0],
                    round(center[1] - triangle_radius),
                ),
                (
                    round(center[0] + triangle_radius * 0.87),
                    round(center[1] + triangle_radius * 0.5),
                ),
                (
                    round(center[0] - triangle_radius * 0.87),
                    round(center[1] + triangle_radius * 0.5),
                ),
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            image,
            [triangle],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        label = "WORSHIP"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(
            0.55,
            face_width * 0.78 / cv2.getTextSize(label, font, 1, thickness)[0][0],
        )
        letter_spacing = max(1, round(font_scale * 2))
        letter_widths = [
            cv2.getTextSize(character, font, font_scale, thickness)[0][0]
            for character in label
        ]
        label_width = sum(letter_widths) + letter_spacing * (len(label) - 1)
        baseline = center[1] - radius - 3
        cursor_x = center[0] - label_width // 2
        for character, character_width in zip(label, letter_widths):
            cv2.putText(
                image,
                character,
                (cursor_x, baseline),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            cursor_x += character_width + letter_spacing
