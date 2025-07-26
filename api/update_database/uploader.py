import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# --- Configuration ---
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("Google Cloud Bucket Name not found in environment variables.")

# --- Initialize Google Cloud Storage ---
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

@router.post("/upload_video/")
async def upload_video(
    video_file: UploadFile = File(...),
    teacher_name: str = Query(..., description="Name of the teacher."),
    grade: str = Query(..., description="Grade level."),
    subject: str = Query(..., description="Subject name.")
):
    """Uploads a video file to Google Cloud Storage with folder structure <teacher_name>/<grade>/<subject>."""
    try:
        # Construct the GCS object path with the desired folder structure
        object_path = f"{teacher_name}/{grade}/{subject}/{video_file.filename}"

        # Upload the file to the specified bucket and path
        blob = bucket.blob(object_path)
        blob.upload_from_file(video_file.file)

        gcs_uri = f"gs://{BUCKET_NAME}/{object_path}"

        # Important: You might want to return the structured GCS URI
        # and potentially other metadata for indexing later.
        return {"message": f"Video '{video_file.filename}' uploaded to '{gcs_uri}'.",
                "gcs_uri": gcs_uri,
                "teacher_name": teacher_name,
                "grade": grade,
                "subject": subject}
    except Exception as e:
        print(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading video: {e}")

@router.post("/upload_pdf/")
async def upload_pdf(
    pdf_file: UploadFile = File(...),
    teacher_name: str = Query(..., description="Name of the teacher."),
    grade: str = Query(..., description="Grade level."),
    subject: str = Query(..., description="Subject name.")
):
    """Uploads a PDF file to Google Cloud Storage with folder structure <teacher_name>/<grade>/<subject>."""
    try:
        # Construct the GCS object path with the desired folder structure
        object_path = f"{teacher_name}/{grade}/{subject}/{pdf_file.filename}"

        # Upload the file to the specified bucket and path
        blob = bucket.blob(object_path)
        blob.upload_from_file(pdf_file.file)

        gcs_uri = f"gs://{BUCKET_NAME}/{object_path}"

        # Important: You might want to return the structured GCS URI
        # and potentially other metadata for indexing later.
        return {"message": f"PDF '{pdf_file.filename}' uploaded to '{gcs_uri}'.",
                "gcs_uri": gcs_uri,
                "teacher_name": teacher_name,
                "grade": grade,
                "subject": subject}
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {e}")

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
        file_list = [blob.name.split("/")[-1] for blob in blobs]
        return {"files": file_list}
    except Exception as e:
        print(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")