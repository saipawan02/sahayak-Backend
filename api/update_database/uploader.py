import os
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from google.cloud import storage
from dotenv import load_dotenv
from typing import List
from api.generate_embedding.embeddings import generate_embedding_task

load_dotenv()

router = APIRouter()

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration ---
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("Google Cloud Bucket Name not found in environment variables.")

# --- Initialize Google Cloud Storage ---
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

@router.post("/upload_files/")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    teacher_name: str = Query(..., description="Name of the teacher."),
    grade: str = Query(..., description="Grade level."),
    subject: str = Query(..., description="Subject name.")
):
    """
    Uploads one or more files to Google Cloud Storage and triggers a background
    task to generate embeddings for each file.
    """
    
    uploaded_files_info = []
    
    for file in files:
        try:
            logger.info(f"Receiving file: {file.filename}")
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in ["mp4", "pdf"]:
                raise HTTPException(status_code=400, detail=f"File type '.{file_extension}' not supported.")

            object_path = f"{teacher_name}/{grade}/{subject}/{file.filename}"
            blob = bucket.blob(object_path)

            file.file.seek(0)
            blob.upload_from_file(file.file)
            gcs_uri = f"gs://{BUCKET_NAME}/{object_path}"
            logger.info(f"Successfully uploaded {file.filename} to {gcs_uri}")

            # Add embedding generation as a background task
            background_tasks.add_task(
                generate_embedding_task,
                teacher=teacher_name,
                grade=grade,
                subject=subject,
                file_name=file.filename
            )
            logger.info(f"Queued embedding generation task for {file.filename}")
            
            uploaded_files_info.append({
                "filename": file.filename,
                "gcs_uri": gcs_uri,
                "content_type": file.content_type,
                "embedding_task_queued": True
            })

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {e}")
            
    return {"uploaded_files": uploaded_files_info}

@router.get("/list_files/")
async def list_files(
    teacher_name: str = Query(..., description="Name of the teacher."),
    grade: str = Query(..., description="Grade level."),
    subject: str = Query(..., description="Subject name.")
):
    """Lists all files in a given directory of the GCS bucket."""
    try:
        prefix = f"{teacher_name}/{grade}/{subject}/"
        blobs = bucket.list_blobs(prefix=prefix)
        file_list = [blob.name for blob in blobs]
        return {"files": file_list}
    except Exception as e:
        logger.error(f"Error listing files for prefix '{prefix}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing files.")