from typing import List

from pydantic import BaseModel, validator

SUPPORTED_SOURCE_IDS = ["care", "marketing", "flow", "cxi"]


class RequestV1(BaseModel):
    source_id: str
    project_id: str
    images: List[dict] = [
        {"image_url": "https://pbs.twimg.com/media/Fgg6SARXgAEYbm0.jpg"}
    ]

    @validator('images')
    def cap_length(cls, v):
        if len(v) > 10:
            raise ValueError(
                "A maximum of 10 images can be handled "
                f"You provided: {len(v)}")
        return v

    @validator('source_id')
    def is_valid_source(cls, v):
        if v not in SUPPORTED_SOURCE_IDS:
            raise ValueError(
                f"Your source {v} is not supported. "
                f"Pick one of: {SUPPORTED_SOURCE_IDS}"
                )
        return v.lower()


class ResponseV1(BaseModel):
    images: List[dict]
    source_id: str
    project_id: str
    # image_url: str
    # generated_caption: str
