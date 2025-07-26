import os
from fastapi import APIRouter, HTTPException
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform_v1, storage, translate
from google.cloud.aiplatform.matching_engine import matching_engine_index_endpoint
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List

load_dotenv()

router = APIRouter()

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
gemini_api_key = os.getenv("GEMINI_API_KEY")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not all([PROJECT_ID, REGION, gemini_api_key, BUCKET_NAME]):
    raise ValueError("Missing required environment variables for Google Cloud, Vertex AI, or Gemini.")

# --- Initialize Clients ---
vertexai.init(project=PROJECT_ID, location=REGION)
genai.configure(api_key=gemini_api_key)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
translate_client = translate.TranslationServiceClient()

# --- Models ---
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
generation_model = genai.GenerativeModel('gemini-pro')

deployed_index_endpoint = None

def retrieve_original_content(item_id: str) -> str:
    print(f"Attempting to retrieve content for item ID: {item_id} from bucket {BUCKET_NAME}")
    object_path = item_id
    try:
        blob = bucket.blob(object_path)
        if blob.exists():
            content = blob.download_as_text()
            return content
        else:
            print(f"Blob not found: {object_path}")
            return f"Content for item ID {item_id}: [Original content not found]"
    except Exception as e:
        print(f"Error retrieving content for item ID {item_id} from GCS: {e}")
        return f"Content for item ID {item_id}: [Error retrieving content]"

@router.post("/translate/")
async def translate_text(strs_to_translate: List[str], target_language_code: str):
    """Translating Text."""
    
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    # Translate text from en to fr
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": strs_to_translate,
            "mime_type": "text/html",  # mime types: text/plain, text/html
            "target_language_code": target_language_code,
        }
    )

    translated_texts = [translation.translated_text for translation in response.translations]

    return translated_texts

@router.post("/generate-content/")
async def generate_content(topic: str):
    """Retrieves information from Vector Search and generates content using Gemini."""
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must be provided.")
    if deployed_index_endpoint is None:
        raise HTTPException(status_code=500, detail="Vertex AI Vector Search endpoint not initialized.")
    try:
        query_embeddings_response = embedding_model.get_embeddings(instances=[{"text": topic}])
        query_vector = query_embeddings_response.text_embeddings[0].embedding
        retrieval_results = deployed_index_endpoint.find_neighbors(
            queries=[query_vector],
            num_neighbors=10
        )
        context = "Retrieved relevant information from knowledge base:"
        if retrieval_results and retrieval_results[0].neighbors:
            for i, neighbor in enumerate(retrieval_results[0].neighbors):
                original_content = retrieve_original_content(neighbor.id)
                context += f"## Source Document/Segment {i+1} (ID: {neighbor.id})"
                context += f"{original_content}"
        else:
            context += "No relevant information found in the knowledge base."
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