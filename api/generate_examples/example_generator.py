import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# --- Initialize Gemini ---
genai.configure(api_key=gemini_api_key)
# Use a current, valid model name
generation_model = genai.GenerativeModel('gemini-2.5-pro')

def generate_examples_from_markdown(markdown_text: str) -> dict:
    """
    Analyzes markdown text and generates a list of relevant examples if needed.
    """
    if not markdown_text:
        return {}

    try:
        prompt = f"""
        Analyze the following markdown text. If and only if the content is complex and would benefit from examples, provide a list of easy-to-understand, highly relevant examples.

        Return a JSON object with a single key, "examples", containing a list of strings. Each string should be a single, concise example.

        If no examples are needed to understand the content, return a JSON object with an empty list: {{"examples": []}}.

        Markdown Text:
        ---
        {markdown_text}
        ---

        Your response must be a single, valid JSON object.
        """
        
        generation_config = {"response_mime_type": "application/json"}

        response = generation_model.generate_content(
            prompt,
            generation_config=generation_config
        )

        example_data = json.loads(response.text)
        return example_data

    except Exception as e:
        print(f"Error during example generation: {e}")
        return {"error": f"An error occurred during example generation: {str(e)}"}
