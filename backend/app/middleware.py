from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from app.config import MAX_REQUEST_SIZE_BYTES

ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_size: int = MAX_REQUEST_SIZE_BYTES) -> None:
        self.app = app
        self.max_size = max_size
        MultiPartParser.spool_max_size = max(
            MultiPartParser.spool_max_size, max_size + 1
        )

    async def __call__(
        self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend
    ) -> None:
        if (
            scope["type"] != "http"
            or scope["path"]
            not in {"/api/v1/face/validate", "/api/v1/face/annotate"}
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_size:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "error": {
                            "code": "FILE_TOO_LARGE",
                            "message": "El archivo supera el tamaño máximo permitido.",
                        },
                    },
                )
                await response(scope, receive, send)
                return
            body.extend(chunk)
            more_body = message.get("more_body", False)

        buffered_body = bytes(body)
        body_sent = False

        async def replay_body() -> ASGIMessage:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": buffered_body,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_body, send)
