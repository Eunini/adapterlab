from fastapi import HTTPException


class BodyLimit:
    def __init__(self, app, limit=8_000_000):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.limit:
                    raise HTTPException(413, "Request body exceeds 8 MB")
            return message

        await self.app(scope, bounded_receive, send)
