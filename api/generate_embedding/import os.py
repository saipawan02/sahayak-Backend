import os
import logging
from fastapi import APIRouter, HTTPException
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video, Image
from google.cloud import storage
from google.cloud import aiplatform # Correct import for higher-level client
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

router = APIRouter()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not all([PROJECT_ID, REGION, BUCKET_NAME]):
    logger.critical("Missing required environment variables. Please ensure GOOGLE_CLOUD_PROJECT_ID, VERTEX_AI_LOCATION, and GOOGLE_CLOUD_BUCKET_NAME are set.")
    raise ValueError("Missing required environment variables.")

# --- Initialize Vertex AI and Google Cloud Storage ---
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION) # Initialize aiplatform client
storage_client = storage.Client()

# Load the multimodal embedding model
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")


def get_multimodal_embeddings_from_video(video_uri: str, interval_sec: int = 16):
    """Generates multimodal embeddings from a video stored in GCS."""
    try:
        video = Video.from_uri(video_uri)
        # video.duration can be None for some videos, default to a large value or handle
        video_duration = int(video.duration) if video.duration is not None else 3600

        embeddings_response = embedding_model.get_embeddings(
            video=video,
            # Use dictionary for video_segment_config, not aiplatform_v1.types.VideoSegmentConfig
            video_segment_config={
                "start_offset_sec": 0,
                "end_offset_sec": video_duration,
                "interval_sec": interval_sec,
            },
        )
        logger.info(f"Generated {len(embeddings_response.video_embeddings)} video embeddings for {video_uri}.")
        return [
            (ve.embedding, f"video_segment_{ve.start_offset_sec}_{ve.end_offset_sec}")
            for ve in embeddings_response.video_embeddings
        ]
    except Exception as e:
        logger.error(f"Error generating multimodal embeddings from video {video_uri}: {e}")
        logger.debug(traceback.format_exc()) # Log full traceback for debugging
        return []


def get_multimodal_embeddings_from_pdf(pdf_uri: str):
    """Generates multimodal embeddings from both text and images in a PDF stored in GCS."""
    embeddings_with_ids = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1023,
        chunk_overlap=200
    )
    try:
        bucket_name, blob_name = pdf_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        pdf_content = blob.download_as_bytes()

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        logger.info(f"Processing PDF {pdf_uri} with {pdf_document.page_count} pages.")

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text().strip()

            # Embed text chunks
            if page_text:
                text_chunks = text_splitter.split_text(page_text)
                for i, chunk in enumerate(text_chunks):
                    try:
                        text_response = embedding_model.get_embeddings(
                            contextual_text=chunk
                        )
                        if text_response.text_embedding:
                            embeddings_with_ids.append(
                                (
                                    text_response.text_embedding,
                                    f"page_{page_num + 1}_text_chunk_{i + 1}",
                                )
                            )
                            logger.debug(
                                f"Generated text embedding for page {page_num + 1}, chunk {i + 1}."
                            )
                        else:
                            logger.warning(f"Text embedding was empty for page {page_num + 1}, chunk {i + 1}.")
                    except Exception as e:
                        logger.error(
                            f"Could not generate text embedding for page {page_num + 1}, chunk {i + 1} in {pdf_uri}: {e}"
                        )
                        logger.debug(traceback.format_exc())

            # Embed images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image(image_bytes=image_bytes)
                    image_response = embedding_model.get_embeddings(image=image)
                    if image_response.image_embedding:
                        embeddings_with_ids.append(
                            (
                                image_response.image_embedding,
                                f"page_{page_num + 1}_image_{img_index + 1}",
                            )
                        )
                        logger.debug(
                            f"Generated image embedding for page {page_num + 1}, image {img_index + 1}."
                        )
                    else:
                        logger.warning(f"Image embedding was empty for page {page_num + 1}, image {img_index + 1}.")
                except Exception as e:
                    logger.error(
                        f"Could not generate image embedding for page {page_num + 1}, image {img_index + 1} in {pdf_uri}: {e}"
                    )
                    logger.debug(traceback.format_exc())

        pdf_document.close()
        logger.info(f"Successfully processed PDF {pdf_uri}. Total embeddings: {len(embeddings_with_ids)}")
        return embeddings_with_ids
    except Exception as e:
        logger.error(f"Error processing PDF from GCS {pdf_uri}: {e}")
        logger.debug(traceback.format_exc())
        return []


def generate_embedding_task(teacher: str, grade: str, subject: str, file_name: str):
    """
    Generates multimodal embeddings for a file and directly upserts them
    into the corresponding Vertex AI Vector Search index.
    """
    if not all([teacher, grade, subject, file_name]):
        raise HTTPException(
            status_code=400,
            detail="teacher, grade, subject, and file_name must be provided.",
        )

    try:
        teacher_clean = teacher.replace(" ", "_").lower()
        object_path = f"{teacher}/{grade}/{subject}/{file_name}"
        file_uri = f"gs://{BUCKET_NAME}/{object_path}"

        file_extension = file_name.split(".")[-1].lower()
        embeddings_with_ids = []

        if file_extension == "mp4":
            embeddings_with_ids = get_multimodal_embeddings_from_video(
                file_uri, interval_sec=16
            )
        elif file_extension == "pdf":
            embeddings_with_ids = get_multimodal_embeddings_from_pdf(file_uri)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Only .mp4 and .pdf are supported.",
            )

        if not embeddings_with_ids:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate any embeddings for {file_uri}. Check logs for details.",
            )

        # 1. Find the corresponding Index Endpoint
        endpoint_display_name = f"{teacher_clean}_{grade}_{subject}_index"
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{endpoint_display_name}"')

        if not indexes:
            logger.error(f"Vector Search Index Endpoint with display name '{endpoint_display_name}' not found for file {file_uri}.")
            raise HTTPException(
                status_code=404,
                detail=f"Vector Search Index Endpoint with display name '{endpoint_display_name}' not found.",
            )
        index_endpoint = indexes[0]
        logger.info(f"Found index endpoint: {index_endpoint.resource_name}")

        # 2. Prepare datapoints for upserting
        datapoints = []
        # Keep sanitization for datapoint_id as it's a string identifier
        file_name_sanitized = "".join(c for c in file_name if c.isalnum() or c in "._-")
        for embedding, unique_id_part in embeddings_with_ids:
            datapoint_id = f"{object_path}::{unique_id_part}",
            datapoints.append(
                # Use aiplatform.MatchingEngineIndex.Datapoint for upserting
                aiplatform.MatchingEngineIndex.Datapoint(
                    datapoint_id=datapoint_id,
                    feature_vector=embedding,
                )
            )

        # 3. Upsert embeddings to the Vector Search index
        logger.info(f"Upserting {len(datapoints)} datapoints to endpoint {index_endpoint.resource_name} for {file_uri}...")
        index_endpoint.upsert_datapoints(datapoints=datapoints)
        logger.info("Upsert operation initiated successfully.")

        return {
            "message": f"Successfully initiated upsert of {len(datapoints)} embeddings to Vector Search.",
            "index_endpoint_name": index_endpoint.resource_name,
            "num_embeddings": len(datapoints),
            "file_uri": file_uri
        }

    except HTTPException as he:
        # Re-raise HTTPExceptions as they are intended responses
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred in /generate_embedding/ for file {file_name}: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {e}",
        )