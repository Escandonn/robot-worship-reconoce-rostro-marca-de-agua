from dataclasses import dataclass


@dataclass(frozen=True)
class FaceResult:
    is_face: bool
    faces_count: int
    message: str
    face_boxes: tuple[tuple[int, int, int, int], ...] = ()
