from app.models.face_result import FaceResult


def face_result_to_response(result: FaceResult) -> dict[str, object]:
    return {
        "success": True,
        "is_face": result.is_face,
        "faces_count": result.faces_count,
        "message": result.message,
    }
