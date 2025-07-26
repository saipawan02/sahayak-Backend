import os
from fastapi import APIRouter, HTTPException
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from google.cloud import aiplatform_v1, storage
from google.cloud.aiplatform.matching_engine import matching_engine_index_endpoint
from dotenv import load_dotenv
import fitz # PyMuPDF
import io

load_dotenv()

router = APIRouter()

# --- Configuration --- 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")

if not all([PROJECT_ID, REGION]):
    raise ValueError("Missing required environment variables for Google Cloud or Vertex AI Location.")

# --- Initialize Vertex AI and Google Cloud Storage ---
vertexai.init(project=PROJECT_ID, location=REGION)
storage_client = storage.Client()

# Load the multimodal embedding model
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

def get_multimodal_embeddings_from_video(video_uri: str, interval_sec: int = 16):
    """
    Generates multimodal embeddings from a video stored in GCS.
    """
    try:
        video = Video.from_uri(video_uri)

        video_duration = int(video.duration) if video.duration is not None else None

        if video_duration is None:
             print("Warning: Video duration not available. Using a default end offset.")
             end_offset = 3600 # Default to 1 hour if duration is unknown
        else:
             end_offset = video_duration

        embeddings_response = embedding_model.get_embeddings(
            video=video,
            video_segment_config=aiplatform_v1.types.VideoSegmentConfig(
                start_offset_sec=0,
                end_offset_sec=end_offset,
                interval_sec=interval_sec
            )
        )
        print(f"Generated {len(embeddings_response.video_embeddings)} video embeddings.")
        # Return embeddings and start/end offsets for creating unique IDs
        return [(ve.embedding, ve.start_offset_sec, ve.end_offset_sec) for ve in embeddings_response.video_embeddings]

    except Exception as e:
        print(f"Error generating multimodal embeddings from video: {e}")
        return []

def get_multimodal_embeddings_from_pdf(pdf_uri: str):
    """
    Generates multimodal embeddings from a PDF stored in GCS.
    Extracts text page by page and generates embeddings.
    """
    pdf_embeddings_with_offsets = []
    try:
        # Parse GCS URI
        from urllib.parse import urlparse
        parsed_uri = urlparse(pdf_uri)
        bucket_name = parsed_uri.netloc
        blob_name = parsed_uri.path.lstrip('/')

        if not bucket_name or not blob_name:
             print(f"Invalid GCS PDF URI: {pdf_uri}")
             return []

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download PDF content into memory
        pdf_content = blob.download_as_bytes()

        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        print(f"Processing PDF with {pdf_document.page_count} pages.")

        # Iterate through pages and generate embeddings
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()

            if not page_text.strip():
                 print(f"Skipping empty page {page_num + 1}.")
                 continue

            try:
                # Generate embedding for the page text
                # The multimodal model can embed text input as well
                embeddings_response = embedding_model.get_embeddings(instances=[{"text": page_text}])
                page_embedding = embeddings_response.text_embeddings[0].embedding

                # Store embedding with page number as offset information
                # Using page_num as start_offset and page_num + 1 as end_offset conceptually
                pdf_embeddings_with_offsets.append((page_embedding, page_num, page_num + 1))
                print(f"Generated embedding for page {page_num + 1}.")

            except Exception as e:
                print(f"Error generating embedding for PDF page {page_num + 1}: {e}")
                # Continue processing other pages even if one fails

        pdf_document.close()
        print(f"Generated {len(pdf_embeddings_with_offsets)} PDF embeddings.")
        return pdf_embeddings_with_offsets

    except Exception as e:
        print(f"Error processing PDF from GCS: {e}")
        return []

@router.post("/generate_embedding/")
async def generate_embedding(teacher: str, grade: str, subject: str, file_uri: str, item_id_prefix: str):
    """Generates multimodal embeddings for a file (video or PDF) and stores them in the appropriate Vertex AI Vector Search index based on teacher, grade, and subject.

    Args:
        teacher: The teacher's name.
        grade: The grade level.
        subject: The subject.
        file_uri: Google Cloud Storage URI of the file (e.g., "gs://your-bucket/your-file.mp4" or "gs://your-bucket/your-document.pdf").
        item_id_prefix: A prefix for the unique ID of each file segment for indexing.
    """
    if not all([teacher, grade, subject, file_uri, item_id_prefix]):
        raise HTTPException(status_code=400, detail="teacher, grade, subject, file_uri, and item_id_prefix must be provided.")

    try:
        # Construct the expected index display name
        index_display_name = f"{teacher}_{grade}_{subject}_index"

        # Find the index with the matching display name
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')

        if not indexes:
            raise HTTPException(status_code=404, detail=f"Index with display name '{index_display_name}' not found. Please create the index first.")

        # Assuming there's only one index with this display name
        index = indexes[0]

        # Find a deployed endpoint for the index
        if not index.deployed_indexes:
             raise HTTPException(status_code=404, detail=f"No deployed endpoint found for index '{index_display_name}'. Please deploy the index first.")

        # Assuming the first deployed index is the one we want to use
        deployed_index_id = index.deployed_indexes[0].id
        index_endpoint_name = index.deployed_indexes[0].index_endpoint

        # Initialize the deployed index endpoint
        deployed_index_endpoint = matching_engine_index_endpoint.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoint_name
        )

        # Determine file type based on extension
        file_extension = os.path.splitext(file_uri)[1].lower()

        embeddings_with_offsets = []

        if file_extension == ".mp4":
            print(f"Processing video file: {file_uri}")
            embeddings_with_offsets = get_multimodal_embeddings_from_video(file_uri, interval_sec=16)
        elif file_extension == ".pdf":
            print(f"Processing PDF file: {file_uri}")
            embeddings_with_offsets = get_multimodal_embeddings_from_pdf(file_uri)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Only .mp4 and .pdf are supported.")

        if not embeddings_with_offsets:
             raise HTTPException(status_code=500, detail=f"Failed to generate embeddings for {file_uri}. No embeddings were generated.")

        # Prepare data for indexing
        datapoints = []
        for i, (embedding, start_offset, end_offset) in enumerate(embeddings_with_offsets):
             # Create a unique ID for each datapoint (file segment or page)
             # Include both offsets for clarity and uniqueness
             datapoint_id = f"{item_id_prefix}_{start_offset}_{end_offset}"
             datapoints.append(
                 aiplatform_v1.IndexDatapoint(
                     datapoint_id=datapoint_id,
                     feature_vector=embedding,
                     # Add restrictions or crowding tags here if needed
                     # For example, add a restriction based on file type or original URI
                     # rest={'uri': file_uri, 'file_type': file_extension}
                 )
             )

        if not datapoints:
             raise HTTPException(status_code=500, detail="No datapoints prepared for indexing.")

        # 3. Upsert embeddings to the index
        print(f"Upserting {len(datapoints)} datapoints to deployed index '{deployed_index_id}'...")
        # The upsert_embeddings method expects a list of IndexDatapoint objects or dicts
        deployed_index_endpoint.upsert_embeddings(datapoints)
        print("Upsert operation initiated.")


        return {"message": f"Embeddings generated and upserted to Vector Search index endpoint '{index_endpoint_name}'.",
                "num_embeddings": len(datapoints),
                "file_type": file_extension,
                "index_display_name": index_display_name}

    except Exception as e:
        print(f"Detailed error in /generate_embedding/: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during embedding generation and upsert: {e}")