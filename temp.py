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
import logging

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

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
generation_model = genai.GenerativeModel('gemini-1.5-pro-preview-0409')


def retrieve_original_content(item_id: str) -> str:
    """
    Retrieves the original content of a file from GCS based on a structured item_id.
    The item_id is expected to be in the format: 'path/to/your/file.pdf::chunk_info'
    """
    try:
        # Split the ID to get the actual GCS object path
        object_path = item_id.split("::")[0]
        logger.info(f"Retrieving content for object path: {object_path}")
        
        blob = bucket.blob(object_path)
        if blob.exists():
            # For simplicity, we download the whole text. 
            # A more advanced implementation could use the chunk_info part of the ID
            # to extract a specific portion of the text.
            content = blob.download_as_text()
            logger.info(f"Successfully retrieved content for {object_path}")
            return content
        else:
            return f"[Content for item ID {item_id} not found. Path {object_path} is invalid.]"
    except Exception as e:
        logger.error(f"Error retrieving content for item ID {item_id}: {e}", exc_info=True)
        return f"[Error retrieving content for item ID {item_id}.]"

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
    image_files: Optional[List[UploadFile]] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """
    Generates a comprehensive document based on a topic, with optional multimodal inputs.
    The index is loaded dynamically based on teacher, grade, and subject.
    """
    if not all([topic, teacher, grade, subject]):
        raise HTTPException(status_code=400, detail="topic, teacher, grade, and subject must be provided.")

    try:
        teacher_clean = teacher.replace(" ", "_").lower()
        endpoint_display_name = f"{teacher_clean}_{grade}_{subject}_index"
        
        logger.info(f"Loading Vector Search endpoint: {endpoint_display_name}")
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if not endpoints:
            raise HTTPException(
                status_code=404,
                detail=f"Vector Search endpoint '{endpoint_display_name}' not found.",
            )
        
        index_endpoint = endpoints[0]
        if not index_endpoint.deployed_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"No indexes are deployed to this endpoint: {endpoint_display_name}"
            )
        deployed_index_id = index_endpoint.deployed_indexes[0].id
        logger.info(f"Found deployed index ID: {deployed_index_id}")

        prompt_parts = [f"Generate a comprehensive document about '{topic}'. "]
        if image_files:
            for image_file in image_files:
                image_data = image_file.file.read()
                prompt_parts.append({"mime_type": image_file.content_type, "data": image_data})
        if audio_file:
            audio_data = audio_file.file.read()
            prompt_parts.append({"mime_type": audio_file.content_type, "data": audio_data})

        logger.info("Generating query embedding...")
        query_embeddings_response = embedding_model.get_embeddings(contextual_text=topic)
        query_vector = query_embeddings_response.text_embedding
        
        logger.info(f"Querying endpoint '{endpoint_display_name}'...")
        retrieval_results = index_endpoint.find_neighbors(
            queries=[query_vector],
            deployed_index_id=deployed_index_id,
            num_neighbors=10
        )
        
        context = "Use the following retrieved information as a knowledge base"
        if retrieval_results and retrieval_results[0]:
            logger.info(f"Retrieved {len(retrieval_results[0])} neighbors.")
            for neighbor in retrieval_results[0]:
                # The neighbor.id now contains the full path to the original file
                original_content = retrieve_original_content(neighbor.id)
                context += f"## Source: {neighbor.id.split('::')[0]}"
                context += f"{original_content}"
        else:
            logger.info("No relevant information found in the vector database.")
            context += "No relevant information was found in the knowledge base."
        
        prompt_parts.append(context)

        logger.info("Generating initial html content...")
        initial_response = generation_model.generate_content(prompt_parts)
        html_content = initial_response.text

        logger.info("Generating charts and examples...")
        chart_data = generate_charts_from_html(html_content)
        example_data = generate_examples_from_html(html_content)

        logger.info("Generating final polished document...")
        final_prompt = f"""
        You are an expert technical writer. Your task is to combine the following pieces of information into a single, cohesive, and well-structured html document.

        Base Content:
        ---
        {html_content}
        ---

        Mermaid.js Charts (integrate where appropriate):
        ---
        {json.dumps(chart_data, indent=2)}
        ---

        Examples (integrate where appropriate):
        ---
        {json.dumps(example_data, indent=2)}
        ---

        Produce a final html document that is well-organized, easy to read, and seamlessly integrates the charts and examples.
        """
        final_response = generation_model.generate_content(final_prompt)
        final_document_en = final_response.text

        final_document = final_document_en
        if language.lower() != "en":
            logger.info(f"Translating final document to {language}...")
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
            logger.info("Translation complete.")

        
        
        return generation_model.generate_content(f"""
        Give me well crafted Html page as an output:
        {final_document}
        """)

    except Exception as e:
        logger.error(f"An unexpected error occurred in generate_content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during content generation.")