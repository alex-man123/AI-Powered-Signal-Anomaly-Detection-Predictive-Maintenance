"""TASK 10.8 -- the one shared error-response schema, mirroring what TASK
10.2-10.7's routes actually return for a 404 (`HTTPException(status_code=404,
detail=str(exc))`) and what TASK 10.7's global handler returns for an
unexpected 500 (`{"detail": "Internal server error"}"`) -- both are already
exactly `{"detail": <str>}`, so ONE schema documents both, not a schema per
endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(min_length=1)
