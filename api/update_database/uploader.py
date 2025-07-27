import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from google.cloud import storage
from dotenv import load_dotenv
from typing import List

load_dotenv()

router = APIRouter()

# --- Configuration ---
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("Google Cloud Bucket Name not found in environment variables.")

# --- Initialize Google Cloud Storage ---
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

@router.post("/upload_files/")
async def upload_files(
    files: List[UploadFile] = File(...),
    teacher_name: str = Query(..., description="Name of the teacher."),
    grade: str = Query(..., description="Grade level."),
    subject: str = Query(..., description="Subject name.")
):
    """Uploads one or more files to Google Cloud Storage, handling them by extension."""
    
    uploaded_files_info = []
    
    for file in files:
        try:
            # Get the file extension
            file_extension = file.filename.split('.')[-1].lower()
            if not file_extension in ["mp4", "pdf"]:
                raise HTTPException(status_code=400, detail=f"File type ._ {file_extension} not supported.")

            # Construct the GCS object path
            object_path = f"{teacher_name}/{grade}/{subject}/{file.filename}"
            blob = bucket.blob(object_path)

            # Upload the file
            file.file.seek(0) # Go to the beginning of the file
            blob.upload_from_file(file.file)

            gcs_uri = f"gs://{BUCKET_NAME}/{object_path}"
            
            uploaded_files_info.append({
                "filename": file.filename,
                "gcs_uri": gcs_uri,
                "content_type": file.content_type,
                "teacher_name": teacher_name,
                "grade": grade,
                "subject": subject
            })

        except Exception as e:
            print(f"Error uploading file {file.filename}: {e}")
            # Continue to next file if one fails, or raise immediately
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
        print(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")