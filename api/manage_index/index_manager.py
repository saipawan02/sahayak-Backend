import os
from fastapi import APIRouter, HTTPException
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# --- Configuration --- 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")

if not all([PROJECT_ID, REGION]):
    raise ValueError("Missing required environment variables for Google Cloud or Vertex AI Location.")

# --- Initialize Vertex AI ---
aiplatform.init(project=PROJECT_ID, location=REGION)

@router.post("/create_index/")
async def create_index(
    display_name: str,
    dimensions: int,
    distance_measure_type: str = "DOT_PRODUCT_DENORMALIZE", # Or "COSINE", "L2_L2"
    contents_delta_uri: str = None, # GCS bucket URI for initial data
    description: str = None
):
    """Initiates the creation of a Vertex AI Vector Search index.

    Args:
        display_name: The display name of the index.
        dimensions: The number of dimensions of the embeddings.
        distance_measure_type: The distance measure type (e.g., "DOT_PRODUCT_DENORMALIZE").
        contents_delta_uri: The GCS bucket URI for the initial data to be indexed.
        description: A description of the index.

    Returns:
        dict: Information about the initiated index creation operation.
    """
    try:
        # Define index configuration
        index_config = aiplatform.MatchingEngineIndexConfig(
            display_name=display_name,
            version_aliases=['default'], # Optional: Add version aliases
            description=description,
            machine_type='e2-highmem-16', # Adjust machine type as needed
            # Specify tree-ah config (required for index creation)
            tree_ah_config=aiplatform.MatchingEngineIndexConfig.TreeAhConfig(
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=10,
            ),
            dimensions=dimensions,
            approximate_neighbors_count=100,
            distance_measure_type=distance_measure_type,
            # Configure embedding storage if providing initial data
            embedding_storage=
                aiplatform.MatchingEngineEmbeddingStorageConfig.create_with_format(
                    aiplatform.MatchingEngineEmbeddingStorageConfig.DataFormat.JSON,
                ) if contents_delta_uri else None,
        )

        # Create the index
        operation = aiplatform.MatchingEngineIndex.create_from_metadata(
            index_config=index_config,
            # Specify contents_delta_uri if providing initial data
            contents_delta_uri=contents_delta_uri if contents_delta_uri else None,
        )

        print(f"Index creation operation initiated: {operation.operation.name}")

        return {"message": "Index creation initiated.", "operation_name": operation.operation.name}

    except Exception as e:
        print(f"Error initiating index creation: {e}")
        raise HTTPException(status_code=500, detail=f"Error initiating index creation: {e}")

@router.get("/check_index_status/")
async def check_index_status(operation_name: str):
    """Checks the status of a Vertex AI Vector Search index creation or update operation.

    Args:
        operation_name: The full operation name (e.g., projects/PROJECT_ID/locations/REGION/operations/OPERATION_ID).

    Returns:
        dict: The status of the operation.
    """
    try:
        # Get the operation
        operation = aiplatform.get_operation(operation_name)

        status = {
            "name": operation.operation.name,
            "done": operation.operation.done,
            "state": operation.operation.metadata.partial_dict().get('generic_metadata', {}).get('state', 'UNKNOWN'),
            "create_time": operation.operation.metadata.create_time.isoformat() if operation.operation.metadata.create_time else None,
            "update_time": operation.operation.update_time.isoformat() if operation.operation.update_time else None,
        }

        if operation.operation.error:
            status["error"] = operation.operation.error.message

        return {"status": status}

    except Exception as e:
        print(f"Error checking operation status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking operation status: {e}")