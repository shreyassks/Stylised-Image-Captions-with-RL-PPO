from fastapi import APIRouter

from app.api.v1.endpoints import generate_captions

v1_router = APIRouter()
v1_router.include_router(
    generate_captions.router,
    prefix="/v1",
)
