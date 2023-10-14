import os
import sys

from fastapi import APIRouter

from app.schemas.schemas import RequestV1, ResponseV1
from app.utils.helper import map_to_caption

router = APIRouter()


# s3 = boto3.client("s3")


# @router.on_event("startup")
# async def startup_event():
#     filename = "app/artifacts/swin_transformer.pth"
#
#     with tqdm.tqdm(total=931000000, unit="B", unit_scale=True, desc=filename) as pbar:
#         s3.download_file(
#             Bucket="flow-neo-dexter",
#             Key="image-captioning/swin_transformer.pth",
#             Filename=filename,
#             Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
#         )


@router.post("/generate_captions")
def generate_captions(request: RequestV1):
    """
    Endpoint to generate captions given an Image

    Returns:
        JSONResponse: Image Link and Generated Caption
    """
    try:
        images_response = list(map(map_to_caption, request.images))
        return ResponseV1(**{
            "source_id": request.source_id,
            "project_id": request.project_id,
            "images": images_response
        })
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error e: {e}")
        print(exc_type, filename, exc_tb.tb_lineno)
