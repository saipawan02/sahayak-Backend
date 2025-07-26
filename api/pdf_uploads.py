import os
from fastapi import APIRouter, File, UploadFile
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

@router.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    """Uploads a PDF file to Google Cloud Storage."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
    service_account_file = os.getenv("GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE")

    if not all([project_id, bucket_name, service_account_file]):
        return {"error": "Google Cloud credentials not found in environment variables."}

    try:
        storage_client = storage.Client.from_service_account_json(service_account_file)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(pdf_file.filename)

        # Upload the file
        blob.upload_from_file(pdf_file.file)

        return {"message": f"PDF '{pdf_file.filename}' uploaded to bucket '{bucket_name}'."}
    except Exception as e:
        return {"error": f"Error uploading PDF: {e}"}