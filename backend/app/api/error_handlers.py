"""TASK 10.7 -- global error handling.

AC1: any UNCONTROLLED exception (one no route/service already turned into an
`HTTPException`) becomes a structured HTTP 500 JSON response -- never a leaked
traceback, internal file path, or other implementation detail -- and is
logged server-side via this project's own existing logging mechanism
(`app.core.logging`, stdlib `logging` -- no new logging framework introduced).

Confirmed BEFORE writing this module (not assumed): FastAPI/Starlette's own
DEFAULT behavior for an unhandled exception is already HTTP 500 with no
traceback leaked to the client -- but the body is `text/plain` ("Internal
Server Error"), not the structured JSON body AC1 requires. This module's only
real job is registering ONE handler that converts that into JSON, in the same
`{"detail": ...}` shape every other error response in this API already uses
(TASK 10.2-10.6's own `HTTPException(status_code=..., detail=...)` calls).

AC2 (Pydantic validation -> HTTP 422 with per-field details) is intentionally
NOT reimplemented here: FastAPI's own default `RequestValidationError` handler
already produces exactly this (`{"detail": [{"type", "loc", "msg", "input",
...}, ...]}`, one entry per invalid field -- verified directly against this
project's own real endpoints, e.g. `POST /api/fft`/`POST /api/dsp/filter`,
before writing this module). Reused UNCHANGED, per this task's own explicit
instruction not to duplicate already-correct behavior -- registering only a
generic `Exception` handler here does not shadow it: Starlette resolves the
MOST SPECIFIC registered handler for a given exception's MRO, and FastAPI
registers its own handler for the more specific `RequestValidationError`
(and for `(Starlette)HTTPException`, which TASK 10.2-10.6's routes already
raise for their own 404/422 cases) ahead of the generic `Exception` handler
registered below -- confirmed directly by this task's own tests, not merely
assumed from Starlette's documented resolution order.

TESTING DETAIL (verified against Starlette 1.6's own source, not assumed):
a handler registered for `Exception` (or the integer `500`) is NOT dispatched
through `ExceptionMiddleware` like every other handler -- Starlette's own
`Starlette.build_middleware_stack()` pulls it out specially and passes it to
`ServerErrorMiddleware` (the outermost layer) as its `handler=`. That
middleware calls the handler, sends the resulting response to the real
client, and then -- by Starlette's own explicit design ("We always continue
to raise the exception. This allows servers to log the error, or allows test
clients to optionally raise the error within the test case.") -- RE-RAISES
the original exception up the ASGI call stack regardless. A real HTTP client
only ever sees the JSON response (the exception never reaches it); FastAPI's
`TestClient` defaults to `raise_server_exceptions=True`, which surfaces that
re-raised exception to the calling test code as a plain Python exception
instead of a response object -- this is a deliberate Starlette testing/
debugging aid, not a bug in this handler. Any test exercising THIS handler's
500 path must therefore construct its `TestClient` with
`raise_server_exceptions=False` to observe the actual HTTP response (TASK
10.2-10.6's existing 404/422 paths are unaffected: those go through
`ExceptionMiddleware`, not `ServerErrorMiddleware`, and never re-raise).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Logs the full exception + traceback server-side (`logger.exception`,
    called while still inside the exception context Starlette invokes this
    handler in, so `sys.exc_info()` is populated) and returns a generic,
    safe, structured JSON body -- no exception message, type, path, or
    traceback is ever included in the response given back to the client."""
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register_error_handlers(app: FastAPI) -> None:
    """Registers this project's global exception handler(s) on `app`. Called
    once from `app.main` at application startup. Only the generic `Exception`
    base class is registered -- see module docstring for why this does not
    shadow FastAPI's own more specific `RequestValidationError`/`HTTPException`
    handling."""
    app.add_exception_handler(Exception, unhandled_exception_handler)
