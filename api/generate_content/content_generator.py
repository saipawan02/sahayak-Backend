import os
from fastapi import APIRouter, HTTPException
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform_v1, storage
from google.cloud.aiplatform.matching_engine import matching_engine_index_endpoint
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# --- Configuration --- 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
VECTOR_STORE_INDEX_ID = os.getenv("VERTEX_AI_INDEX_ID")
VECTOR_STORE_ENDPOINT_ID = os.getenv("VERTEX_AI_ENDPOINT_ID")
DEPLOYED_INDEX_ID = VECTOR_STORE_ENDPOINT_ID
gemini_api_key = os.getenv("GEMINI_API_KEY")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME") # Get bucket name for retrieval

if not all([PROJECT_ID, REGION, VECTOR_STORE_INDEX_ID, VECTOR_STORE_ENDPOINT_ID, gemini_api_key, BUCKET_NAME]):
    raise ValueError("Missing required environment variables for Google Cloud, Vertex AI, or Gemini.")

# --- Initialize Vertex AI, Gemini, and Google Cloud Storage ---
vertexai.init(project=PROJECT_ID, location=REGION)
genai.configure(api_key=gemini_api_key)
storage_client = storage.Client() # Initialize storage client for retrieval
bucket = storage_client.bucket(BUCKET_NAME) # Get bucket for retrieval

# Get the deployed index endpoint
try:
    deployed_index_endpoint = matching_engine_index_endpoint.MatchingEngineIndexEndpoint(
        index_endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{VECTOR_STORE_ENDPOINT_ID}"
    )
except Exception as e:
    print(f"Error initializing MatchingEngineIndexEndpoint: {e}")
    deployed_index_endpoint = None # Handle case where endpoint initialization fails

# Load the multimodal embedding model for query embedding
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Load the Gemini model
generation_model = genai.GenerativeModel('gemini-pro')

# Updated Placeholder function to retrieve original content based on item ID
# THIS FUNCTION NEEDS TO BE FULLY IMPLEMENTED BY YOU
def retrieve_original_content(item_id: str) -> str:
    """Retrieves the original content (text, summary, etc.) for a given item ID from GCS.

    This function assumes item_ids are structured in a way that allows reconstructing the GCS path.
    For example, if item_id is like "teacher/grade/subject/filename_segment_start_end",
    you need to parse this to get the bucket name and object path.
    """
    print(f"Attempting to retrieve content for item ID: {item_id} from bucket {BUCKET_NAME}")
    # --- YOUR IMPLEMENTATION GOES HERE ---
    # This is a simplified example assuming item_id directly or indirectly provides the GCS path.
    # A robust implementation would need a mapping from item_id to the original GCS URI
    # or parse the item_id carefully if the URI information is encoded within it.

    # Example (Conceptual): Assuming item_id is the full object path within the bucket
    object_path = item_id # This is a placeholder assumption - adjust based on your item_id format

    try:
        blob = bucket.blob(object_path)
        if blob.exists():
            # Read content based on file type (infer from object_path or stored metadata)
            # For a simple example, let's assume we can read it as text (suitable for PDF text)
            # For videos, you might have stored a transcript or summary and retrieve that.
            # You'll need more sophisticated logic here based on your data.
            content = blob.download_as_text() # Or download_as_bytes() and process
            return content
        else:
            print(f"Blob not found: {object_path}")
            return f"Content for item ID {item_id}: [Original content not found]"

    except Exception as e:
        print(f"Error retrieving content for item ID {item_id} from GCS: {e}")
        return f"Content for item ID {item_id}: [Error retrieving content]"


@router.post("/generate-content/")
async def generate_content(topic: str):
    """Retrieves information from Vector Search and generates content using Gemini."""
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must be provided.")

    if deployed_index_endpoint is None:
         raise HTTPException(status_code=500, detail="Vertex AI Vector Search endpoint not initialized.")

    try:
        # 1. Generate embedding for the topic
        query_embeddings_response = embedding_model.get_embeddings(instances=[{"text": topic}])
        query_vector = query_embeddings_response.text_embeddings[0].embedding # Access text_embeddings for text input

        # 2. Retrieve information from Vector Search
        retrieval_results = deployed_index_endpoint.find_neighbors(
            queries=[query_vector],
            num_neighbors=10 # Retrieve top 10 neighbors
        )

        # 3. Format retrieved information as context for Gemini
        context = "Retrieved relevant information from knowledge base:"
        if retrieval_results and retrieval_results[0].neighbors:
            for i, neighbor in enumerate(retrieval_results[0].neighbors):
                # Fetch original content using the item ID
                original_content = retrieve_original_content(neighbor.id)
                context += f"## Source Document/Segment {i+1} (ID: {neighbor.id})"
                context += f"{original_content}"
        else:
            context += "No relevant information found in the knowledge base."

        # 4. Generate content with Gemini
        prompt = f"""Generate a comprehensive Markdown document about '{topic}' based on the following information. 

{context}

Structure the document with headings, bullet points, and clear explanations. Ensure the content is directly supported by the provided information."""

        response = generation_model.generate_content(prompt)

        if response.candidates and response.candidates[0].finish_reason == 1:
             raise HTTPException(status_code=400, detail="Content generation blocked due to safety policy.")

        return {"markdown_content": response.text}

    except Exception as e:
        print(f"Detailed error in /generate-content/: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {e}")