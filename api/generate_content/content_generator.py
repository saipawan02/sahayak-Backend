import os
from fastapi import APIRouter, HTTPException
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform_v1
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

if not all([PROJECT_ID, REGION, VECTOR_STORE_INDEX_ID, VECTOR_STORE_ENDPOINT_ID, gemini_api_key]):
    raise ValueError("Missing required environment variables for Google Cloud, Vertex AI, or Gemini.")

# --- Initialize Vertex AI and Gemini ---
vertexai.init(project=PROJECT_ID, location=REGION)
genai.configure(api_key=gemini_api_key)

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

# Placeholder function to retrieve original content based on item ID
# THIS FUNCTION NEEDS TO BE FULLY IMPLEMENTED BY YOU
def retrieve_original_content(item_id: str) -> str:
    """Retrieves the original content (text, summary, etc.) for a given item ID.

    This function should handle different types of item IDs (e.g., video segments, PDF pages).
    It should fetch the relevant content from your data source (e.g., GCS, database).
    """
    print(f"Attempting to retrieve content for item ID: {item_id}")
    # --- YOUR IMPLEMENTATION GOES HERE ---
    # Example: If item_id is like "video_segment_16", fetch content for the video segment starting at 16 seconds.
    # Example: If item_id is like "pdf_page_3", fetch content for page 3 of the PDF.
    # The return value should be a string containing the relevant information to be used as context.
    return f"Content for item ID {item_id}: [Placeholder - Implement actual content retrieval]"

@router.post("/generate-content/")
async def generate_content(topic: str):
    """Retrieves information from Vector Search and generates content using Gemini."""
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must be provided.")

    if deployed_index_endpoint is None:
         raise HTTPException(status_code=500, detail="Vertex AI Vector Search endpoint not initialized.")

    try:
        # 1. Generate embedding for the topic
        # Using MultiModalEmbeddingModel directly for text embedding
        query_embeddings_response = embedding_model.get_embeddings(instances=[{"text": topic}])
        query_vector = query_embeddings_response.text_embeddings[0].embedding # Access text_embeddings for text input

        # 2. Retrieve information from Vector Search
        # Adjust number of neighbors and other parameters as needed
        retrieval_results = deployed_index_endpoint.find_neighbors(
            queries=[query_vector],
            num_neighbors=10 # Retrieve top 10 neighbors
        )

        # 3. Format retrieved information as context for Gemini
        context = "Retrieved relevant information from knowledge base:"
        if retrieval_results and retrieval_results[0].neighbors:
            for i, neighbor in enumerate(retrieval_results[0].neighbors):
                # Fetch original content using the item ID (including segment info for videos)
                original_content = retrieve_original_content(neighbor.id)
                context += f"## Source Document/Segment {i+1} (ID: {neighbor.id})"
                context += f"{original_content}"
        else:
            context += "No relevant information found in the knowledge base."

        # 4. Generate content with Gemini
        # Suggestion for Prompt 1: Direct Generation with Context
        prompt1 = f"""Generate a comprehensive Markdown document about '{topic}' based on the following information. 

{context}

Structure the document with headings, bullet points, and clear explanations. Ensure the content is directly supported by the provided information."""

        # Suggestion for Prompt 2: Q&A Style Generation
        # This prompt is useful if the retrieved content contains Q&A pairs or specific facts.
        prompt2 = f"""Answer the question '{topic}' using only the following information. Respond in Markdown format. If the information does not contain the answer, state that you cannot answer based on the provided context.

{context}"""

        # Suggestion for Prompt 3: Summarization and Synthesis
        # This prompt encourages Gemini to synthesize information from multiple sources.
        prompt3 = f"""Synthesize the key information from the following sources to create a Markdown summary about '{topic}'. Focus on the most important points and present them concisely.

{context}"""

        # Choose which prompt to use. For a general RAG system, Prompt 1 is often a good starting point.
        chosen_prompt = prompt1

        response = generation_model.generate_content(chosen_prompt)

        # Check if the response is blocked due to safety reasons
        if response.candidates and response.candidates[0].finish_reason == 1:
             raise HTTPException(status_code=400, detail="Content generation blocked due to safety policy.")

        return {"markdown_content": response.text}

    except Exception as e:
        # Log the detailed error on the server side
        print(f"Detailed error in /generate-content/: {e}")
        # Return a more generic error to the client
        raise HTTPException(status_code=500, detail="An error occurred during content generation.")