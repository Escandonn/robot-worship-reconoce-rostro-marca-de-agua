import os


def _positive_integer_from_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error

    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


MAX_UPLOAD_SIZE_BYTES = _positive_integer_from_environment(
    "MAX_UPLOAD_SIZE_BYTES", 5 * 1024 * 1024
)
MAX_REQUEST_SIZE_BYTES = MAX_UPLOAD_SIZE_BYTES + 64 * 1024
