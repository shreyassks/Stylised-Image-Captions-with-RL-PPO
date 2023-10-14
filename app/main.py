from app.api.root import root_router
from app.api.v1.api import v1_router
from app.core.config import get_application

app = get_application()
app.include_router(root_router, tags=["Default"])
app.include_router(v1_router, tags=["V1"])

# if __name__ == "__main__":
#     uvicorn.run("main:app", host='0.0.0.0', proxy_headers=True,
#                 port=int(os.getenv('PORT', 8080)), workers=int(1))
