import os
from fastapi import APIRouter, HTTPException, File, UploadFile
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform_v1, storage, translate, aiplatform
from google.cloud.aiplatform.matching_engine import matching_engine_index_endpoint
import google.generativeai as genai
from dotenv import load_dotenv
import json
from typing import List, Optional
from api.generate_charts.chart_generator import generate_charts_from_html
from api.generate_examples.example_generator import generate_examples_from_html
import traceback

load_dotenv()

router = APIRouter()

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
gemini_api_key = os.getenv("GEMINI_API_KEY")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not all([PROJECT_ID, REGION, gemini_api_key, BUCKET_NAME]):
    raise ValueError("Missing required environment variables.")

# --- Initialize Clients ---
vertexai.init(project=PROJECT_ID, location=REGION)
genai.configure(api_key=gemini_api_key)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
translate_client = translate.TranslationServiceClient()

# --- Models ---
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
generation_model = genai.GenerativeModel('gemini-2.0-flash')


def retrieve_original_content(item_id: str) -> str:
    print(f"Attempting to retrieve content for item ID: {item_id} from bucket {BUCKET_NAME}")
    object_path = item_id
    try:
        blob = bucket.blob(object_path)
        if blob.exists():
            content = blob.download_as_text()
            return content
        else:
            return f"Content for item ID {item_id}: [Original content not found]"
    except Exception as e:
        print(f"Error retrieving content for item ID {item_id} from GCS: {e}")
        return f"Content for item ID {item_id}: [Error retrieving content]"

@router.post("/translate/")
async def translate_text(strs_to_translate: List[str], target_language_code: str):
    """Translating Text."""
    
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": strs_to_translate,
            "mime_type": "text/html",
            "target_language_code": target_language_code,
        }
    )

    translated_texts = [translation.translated_text for translation in response.translations]

    return translated_texts
    
@router.post("/generate-content/")
async def generate_content(
    topic: str,
    teacher: str,
    grade: str,
    subject: str,
    language: str = "en",
    image_file: Optional[UploadFile] | None = None,
    audio_file: Optional[UploadFile] | None = None,
):
    """
    Generates a comprehensive document based on a topic, with optional multimodal inputs.
    The index is loaded dynamically based on teacher, grade, and subject.
    """
    if not all([topic, teacher, grade, subject]):
        raise HTTPException(status_code=400, detail="topic, teacher, grade, and subject must be provided.")

    try:
        # 1. Load the index dynamically
        teacher_clean = teacher.replace(" ", "_").lower()
        endpoint_display_name = f"{teacher_clean}_{grade}_{subject}_index"
        
        # Use aiplatform.MatchingEngineIndexEndpoint.list to find the endpoint
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if not endpoints:
            raise HTTPException(
                status_code=404,
                detail=f"Vector Search endpoint '{endpoint_display_name}' not found. Please create it first.",
            )
        
        # Get the resource name of the endpoint
        endpoint_resource_name = endpoints[0].resource_name
        print(f"Found endpoint resource name: {endpoint_resource_name}")

        # Initialize the MatchingEngineIndexEndpoint with the resource name
        # Set public_endpoint=True if it's a public endpoint
        # If it's a private endpoint, ensure your networking is correctly configured
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_resource_name) 

        if not index_endpoint.deployed_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"No indexes are deployed to this endpoint: {endpoint_display_name}"
            )
        deployed_index_id = index_endpoint.deployed_indexes[0].id
        print(f"Using deployed index ID: {deployed_index_id}")

        # 2. Prepare multimodal prompt
        prompt_parts = [f"Generate a comprehensive document about '{topic}'. "]

        if image_file:
            image_file.file.seek(0)
            image_data = image_file.file.read()
            prompt_parts.append({"mime_type": image_file.content_type, "data": image_data})

        if audio_file:
            audio_file.file.seek(0)
            audio_data = audio_file.file.read()
            prompt_parts.append({"mime_type": audio_file.content_type, "data": audio_data})


        # 3. Retrieve information from Vector Search
        query_embeddings_response = embedding_model.get_embeddings(contextual_text=topic)
        query_vector = query_embeddings_response.text_embedding
        
        retrieval_results = index_endpoint.find_neighbors(
            queries=[query_vector],
            deployed_index_id=deployed_index_id,
            num_neighbors=10
        )
        print(retrieval_results)
        context = "Use the following retrieved information as a knowledge base:"
        if retrieval_results and retrieval_results[0]:
            for i, neighbor in enumerate(retrieval_results[0]):
                original_content = retrieve_original_content(neighbor.id)
                context += f"## Source Document/Segment {i+1} (ID: {neighbor.id})"
                context += f"{original_content}"
        else:
            context += "No relevant information was found in the knowledge base."
        
        prompt_parts.append(context)

        # 4. Generate initial content
        initial_response = generation_model.generate_content(prompt_parts)
        markdown_content = initial_response.text

        # 5. Generate charts and examples
        chart_data = generate_charts_from_markdown(markdown_content)
        example_data = generate_examples_from_markdown(markdown_content)

        # 6. Generate the final, polished document in English
        final_prompt = f"""
        You are an expert technical writer. Your task is to combine the following pieces of information into a single, cohesive, and well-structured markdown document.

        Base Content:
        ---
        {markdown_content}
        ---

        Mermaid.js Charts (integrate where appropriate):
        ---
        {json.dumps(chart_data, indent=2)}
        ---

        Examples (integrate where appropriate):
        ---
        {json.dumps(example_data, indent=2)}
        ---

        Produce a final markdown document that is well-organized, easy to read, and seamlessly integrates the charts and examples.
        """
        final_response = generation_model.generate_content(final_prompt)
        final_document_en = final_response.text

        # 7. Translate the final document if a different language is requested
        final_document = final_document_en
        if language.lower() != "en":
            parent = f"projects/{PROJECT_ID}/locations/{REGION}"
            trans_response = translate_client.translate_text(
                request={
                    "parent": parent,
                    "contents": [final_document_en],
                    "mime_type": "text/plain",
                    "target_language_code": language,
                }
            )
            final_document = trans_response.translations[0].translated_text

        return final_document

    except Exception as e:
        print(f"Detailed error in /generate-content/: {e}")
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")