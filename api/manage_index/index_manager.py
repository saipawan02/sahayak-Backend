import os
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from google.cloud import aiplatform
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
import logging

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Configuration --- 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

if not all([PROJECT_ID, REGION, BUCKET_NAME]):
    raise ValueError("Missing required environment variables for Google Cloud, Vertex AI Location, or Bucket Name.")

# --- Initialize Vertex AI ---
aiplatform.init(project=PROJECT_ID, location=REGION)


def _create_and_deploy_index_task(
    teacher: str,
    grade: str,
    subject: str,
    dimensions: int,
    distance_measure_type: str,
    description: str,
    endpoint_display_name: str
):
    """Background task to create an empty index for streaming and deploy it."""
    try:
        index_display_name = f"{teacher.replace(' ', '_').lower()}_{grade}_{subject}_index"
        
        logger.info(f"Background task: Checking for existing index: {index_display_name}")
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')
        
        if indexes:
            index = indexes[0]
            logger.info(f"Background task: Index already exists: {index.resource_name}")
        else:
            logger.info(f"Background task: Creating new index '{index_display_name}' for stream updates.")
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=index_display_name,
                # Create an empty index, contents_delta_uri is not needed for STREAM_UPDATE
                contents_delta_uri=None, 
                description=description or "Index for real-time updates",
                dimensions=dimensions,
                approximate_neighbors_count=150,
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=7,
                index_update_method="STREAM_UPDATE",
                distance_measure_type=distance_measure_type
            )
            logger.info(f"Background task: Index creation operation initiated for {index_display_name}")

        # --- Endpoint and Deployment Logic (remains the same) ---
        logger.info(f"Finding or creating index endpoint: {endpoint_display_name}")
        endpoints = aiplatform.IndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if endpoints:
            index_endpoint = endpoints[0]
            logger.info(f"Found existing index endpoint: {index_endpoint.resource_name}")
        else:
            logger.info(f"Creating new index endpoint: {endpoint_display_name}")
            index_endpoint = aiplatform.IndexEndpoint.create(
                display_name=endpoint_display_name,
                public_endpoint_enabled=True,
            )
            logger.info(f"Index endpoint created: {index_endpoint.resource_name}")

        deployed_index_id = f"{teacher.replace(' ', '_').lower()}_{grade}_{subject}_deployed_index"
        is_deployed = any(
            deployed.id == deployed_index_id for deployed in index_endpoint.deployed_indexes
        )

        if is_deployed:
            logger.info(f"Index '{index.display_name}' is already deployed to endpoint '{endpoint_display_name}'. Skipping.")
        else:
            logger.info(f"Deploying index '{index.resource_name}' to endpoint '{index_endpoint.resource_name}'...")
            index_endpoint.deploy_index(
                index=index,
                deployed_index_id=deployed_index_id,
                machine_type="e2-highmem-16",
                min_replica_count=1,
                max_replica_count=1,
            )
            logger.info("Index deployment initiated.")

        logger.info(f"Task complete for index '{index_display_name}' and endpoint '{endpoint_display_name}'.")

    except Exception as e:
        logger.error(f"Error during automated index creation and deployment: {e}", exc_info=True)


@router.post("/create_index/")
async def create_index(
    background_tasks: BackgroundTasks,
    teacher: str,
    grade: str,
    subject: str,
    dimensions: int = 1408,
    distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
    description: str = None,
):
    """
    Initiates the creation and deployment of a Vertex AI Vector Search index 
    configured for real-time streaming updates.
    """
    teacher_clean = teacher.replace(' ', '_').lower()
    endpoint_display_name = f"{teacher_clean}_{grade}_{subject}_index"

    background_tasks.add_task(
        _create_and_deploy_index_task,
        teacher,
        grade,
        subject,
        dimensions,
        distance_measure_type,
        description,
        endpoint_display_name
    )

    return {"message": "Index creation and deployment for streaming initiated. Use status checks to monitor progress."}
    
# --- Status Check Endpoints (remain the same) ---
@router.get("/check_index_status/")
async def check_index_status(teacher: str, grade: str, subject: str):
    """Checks the status of a Vertex AI Vector Search index."""
    try:
        display_name = f"{teacher.replace(' ', '_').lower()}_{grade}_{subject}_index"
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{display_name}"')
        if not indexes:
            raise HTTPException(status_code=404, detail=f"Index '{display_name}' not found.")
        return {"status": indexes[0].to_dict()}
    except Exception as e:
        logger.error(f"Error checking index status for '{display_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500)

@router.get("/check_index_endpoint_status/")
async def check_index_endpoint_status(teacher: str, grade: str, subject: str):
    """Checks the status of a Vertex AI Vector Search Index Endpoint."""
    try:
        display_name = f"{teacher.replace(' ', '_').lower()}_{grade}_{subject}_index"
        endpoints = aiplatform.IndexEndpoint.list(filter=f'display_name="{display_name}"')
        if not endpoints:
            raise HTTPException(status_code=404, detail=f"Endpoint '{display_name}' not found.")
        return {"status": endpoints[0].to_dict()}
    except Exception as e:
        logger.error(f"Error checking endpoint status for '{display_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500)