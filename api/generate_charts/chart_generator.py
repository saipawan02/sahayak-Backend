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

def generate_charts_from_markdown(markdown_text: str) -> dict:
    """
    Analyzes markdown text and generates a JSON object with Mermaid.js commands if charts are suitable.
    """
    if not markdown_text:
        return {}

    try:
        prompt = f"""
        Analyze the following markdown text. Identify all opportunities to visualize information as a flowchart, pie chart, or another type of diagram using Mermaid.js.

        Return a JSON object where each key is "chartX" (e.g., "chart1", "chart2") and the value is the complete Mermaid.js command for that chart.

        If no charts are suitable for the given text, return an empty JSON object {{}}.

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

        chart_data = json.loads(response.text)
        return chart_data

    except Exception as e:
        print(f"Error during chart generation: {e}")
        return {"error": f"An error occurred during chart generation: {str(e)}"}
