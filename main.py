from fastapi import FastAPI
from api.update_database import uploader
from api.generate_content import content_generator
from api.manage_index import index_manager
from dotenv import load_dotenv
import os

# Only load the .env file if not running in a Google Cloud Run environment
if "K_SERVICE" not in os.environ:
    load_dotenv()

app = FastAPI()

app.include_router(uploader.router, prefix="/update_database")
app.include_router(content_generator.router, prefix="/generate_content")
app.include_router(index_manager.router, prefix="/manage_index")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
