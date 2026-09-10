from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.error_handlers import register_error_handlers
from app.api.routes.datasets import router as datasets_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.features import router as features_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.signals import router as signals_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="AI-Powered Vibration Signal Anomaly Detection API")

register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(datasets_router, prefix="/api", tags=["Datasets"])
app.include_router(signals_router, prefix="/api", tags=["Signal Processing"])
app.include_router(features_router, prefix="/api", tags=["Features"])
app.include_router(models_router, prefix="/api", tags=["Models"])
app.include_router(experiments_router, prefix="/api", tags=["Experiments"])


def _custom_openapi() -> dict:
    """TASK 10.8: the ONE real 500 contract (TASK 10.7's global handler,
    `{"detail": "Internal server error"}` for literally any endpoint) is
    injected into every operation's documented responses HERE, once -- rather
    than repeating an identical `responses={500: ...}` on every one of Phase
    10's dozen route decorators, which the task's own instructions warn
    against ("nu adăuga descrieri... doar pentru a umple"). This does not
    change any endpoint's actual behavior, only what `/openapi.json` states.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    schema.setdefault("components", {}).setdefault("schemas", {})["ErrorResponse"] = {
        "type": "object",
        "title": "ErrorResponse",
        "properties": {"detail": {"type": "string", "title": "Detail"}},
        "required": ["detail"],
    }
    error_response_ref = {
        "description": "Unexpected server error -- see TASK 10.7's global handler. "
        "Never a stack trace or internal detail.",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
    }
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            operation.setdefault("responses", {}).setdefault("500", error_response_ref)

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi
