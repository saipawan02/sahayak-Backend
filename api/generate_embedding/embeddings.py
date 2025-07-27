import os
import traceback
import json
import logging

import fitz  # PyMuPDF
from fastapi import HTTPException
from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine import matching_engine_index_endpoint
from google.cloud import aiplatform_v1
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video, Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not all([PROJECT_ID, REGION, BUCKET_NAME]):
    raise ValueError("Missing required environment variables.")

# --- Initialize Vertex AI and Google Cloud Storage ---
vertexai.init(project=PROJECT_ID, location=REGION)
storage_client = storage.Client()
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")


def get_multimodal_embeddings_from_video(video_uri: str, interval_sec: int = 16):
    """Generates multimodal embeddings from a video stored in GCS."""
    try:
        logger.info(f"Starting video embedding generation for {video_uri}")
        video = Video.from_uri(video_uri)
        video_duration = int(video.duration) if video.duration is not None else 3600

        embeddings_response = embedding_model.get_embeddings(
            video=video,
            video_segment_config={"interval_seconds": interval_sec},
        )
        logger.info(f"Generated {len(embeddings_response.video_embeddings)} video embeddings for {video_uri}.")
        return [
            (ve.embedding, f"video_segment_{ve.start_offset_sec}_{ve.end_offset_sec}")
            for ve in embeddings_response.video_embeddings
        ]
    except Exception as e:
        logger.error(f"Error generating video embeddings for {video_uri}: {e}")
        logger.debug(traceback.format_exc())
        return []


def get_multimodal_embeddings_from_pdf(pdf_uri: str):
    """Generates multimodal embeddings from both text and images in a PDF stored in GCS."""
    embeddings_with_ids = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1023, chunk_overlap=200)
    try:
        logger.info(f"Starting PDF processing for {pdf_uri}")
        bucket_name, blob_name = pdf_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        pdf_content = blob.download_as_bytes()

        with fitz.open(stream=pdf_content, filetype="pdf") as pdf_document:
            logger.info(f"Processing PDF {pdf_uri} with {pdf_document.page_count} pages.")
            for page_num, page in enumerate(pdf_document):
                page_text = page.get_text().strip()

                if page_text:
                    text_chunks = text_splitter.split_text(page_text)
                    for i, chunk in enumerate(text_chunks):
                        try:
                            text_response = embedding_model.get_embeddings(contextual_text=chunk)
                            if text_response.text_embedding:
                                embeddings_with_ids.append(
                                    (text_response.text_embedding, f"page_{page_num + 1}_text_chunk_{i + 1}")
                                )
                        except Exception as e:
                            logger.warning(f"Could not get text embedding for page {page_num + 1}, chunk {i + 1} of {pdf_uri}: {e}")

                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image(image_bytes=image_bytes)
                        image_response = embedding_model.get_embeddings(image=image)
                        if image_response.image_embedding:
                            embeddings_with_ids.append(
                                (image_response.image_embedding, f"page_{page_num + 1}_image_{img_index + 1}")
                            )
                    except Exception as e:
                        logger.warning(f"Could not get image embedding for page {page_num + 1}, image {img_index + 1} of {pdf_uri}: {e}")
        
        logger.info(f"Successfully processed PDF {pdf_uri}. Total embeddings: {len(embeddings_with_ids)}")
        return embeddings_with_ids
    except Exception as e:
        logger.error(f"Error processing PDF from GCS {pdf_uri}: {e}")
        logger.debug(traceback.format_exc())
        return []


def generate_embedding_task(teacher: str, grade: str, subject: str, file_name: str):
    """
    Background task to generate and upsert embeddings for a file.
    """
    logger.info(f"Starting embedding task for: {teacher}/{grade}/{subject}/{file_name}")
    try:
        teacher_clean = teacher.replace(" ", "_").lower()
        object_path = f"{teacher_clean}/{grade}/{subject}/{file_name}"
        file_uri = f"gs://{BUCKET_NAME}/{object_path}"

        file_extension = file_name.split(".")[-1].lower()
        embeddings_with_ids = []

        if file_extension == "mp4":
            embeddings_with_ids = get_multimodal_embeddings_from_video(file_uri)
        elif file_extension == "pdf":
            embeddings_with_ids = get_multimodal_embeddings_from_pdf(file_uri)
        else:
            logger.warning(f"Unsupported file type '{file_extension}' for {file_uri}. Skipping embedding.")
            return

        if not embeddings_with_ids:
            logger.warning(f"No embeddings generated for {file_uri}. Aborting task.")
            return

        endpoint_display_name = f"{teacher_clean}_{grade}_{subject}_index"
        endpoints = aiplatform.IndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if not endpoints:
            logger.error(f"Endpoint '{endpoint_display_name}' not found. Cannot upsert embeddings for {file_uri}.")
            return

        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoints[0].resource_name)
        if not index_endpoint.deployed_indexes:
            logger.error(f"No deployed indexes on endpoint '{endpoint_display_name}'. Cannot upsert for {file_uri}.")
            return
        
        deployed_index_id = index_endpoint.deployed_indexes[0].id

        file_name_sanitized = "".join(c for c in file_name if c.isalnum() or c in "._-")
        datapoints = [
            aiplatform_v1.IndexDatapoint(
                datapoint_id=f"{file_name_sanitized}_{unique_id_part}",
                feature_vector=embedding,
                restricts=[aiplatform_v1.Restriction(namespace="file_uri", allow_list=[file_uri])],
            )
            for embedding, unique_id_part in embeddings_with_ids
        ]

        logger.info(f"Upserting {len(datapoints)} datapoints for {file_uri} to endpoint '{endpoint_display_name}'...")
        index_endpoint.upsert_datapoints(datapoints=datapoints, deployed_index_id=deployed_index_id)
        logger.info(f"Successfully initiated upsert for {file_uri}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in background task for {file_name}: {e}")
        logger.error(traceback.format_exc())
