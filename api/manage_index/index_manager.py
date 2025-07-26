import os
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from google.cloud import aiplatform
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

load_dotenv()

router = APIRouter()

# --- Configuration --- 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("VERTEX_AI_LOCATION")

if not all([PROJECT_ID, REGION]):
    raise ValueError("Missing required environment variables for Google Cloud or Vertex AI Location.")

# --- Initialize Vertex AI ---
aiplatform.init(project=PROJECT_ID, location=REGION)

def _create_and_deploy_index_task(
    teacher: str,
    grade: str,
    subject: str,
    dimensions: int,
    distance_measure_type: str,
    contents_delta_uri: str,
    description: str,
    endpoint_display_name: str
):
    """Background task to create index and deploy it to an endpoint."""
    try:
        index_display_name = f"{teacher}_{grade}_{subject}_index"
        
        # 1. Check if the index already exists
        print(f"Background task: Checking for existing index: {index_display_name}")
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')
        
        if indexes:
            index = indexes[0]
            print(f"Background task: Index already exists: {index.resource_name}")
        else:
            # Create the index if it doesn't exist
            print(f"Background task: Initiating creation of index: {index_display_name}")
            index_operation = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=index_display_name,
                contents_delta_uri=contents_delta_uri,
                description=description or "Matching Engine Index",
                dimensions=dimensions,
                approximate_neighbors_count=150,
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=7,
                index_update_method="BATCH_UPDATE",
                distance_measure_type=distance_measure_type
            )

            # Wait for index creation to complete
            print("Background task: Waiting for index creation to complete...")
            index = index_operation.result()
            print(f"Background task: Index created: {index.resource_name}")


        # 2. Find or create the index endpoint
        print(f"Background task: Finding or creating index endpoint: {endpoint_display_name}")
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if endpoints:
            index_endpoint = endpoints[0]
            print(f"Background task: Found existing index endpoint: {index_endpoint.resource_name}")
        else:
            print(f"Background task: Creating new index endpoint: {endpoint_display_name}")
            endpoint_operation = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=endpoint_display_name,
                public_endpoint_enabled=True,
            )


            while True:
                index_endpoints = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')
                if index_endpoint is not None:
                    index_endpoint = index_endpoint[0]
                    break
                print("Background task: Waiting for index to be created...")
                time.sleep(5)
            
            print(f"Background task: Index endpoint created: {index_endpoint.resource_name}")

        # 3. Deploy the index to the endpoint
        deployed_index_id = f"{teacher.replace(' ', '_').lower()}_{grade}_{subject}_deployed_index"
        
        is_deployed = any(
            deployed.id == deployed_index_id for deployed in index_endpoint.deployed_indexes
        )

        if is_deployed:
            print(f"Background task: Index '{index.display_name}' is already deployed to endpoint '{index_endpoint.display_name}' with ID '{deployed_index_id}'. Skipping deployment.")
        else:
            print(f"Background task: Initiating deployment of index '{index.resource_name}' to endpoint '{index_endpoint.resource_name}' with deployed ID '{deployed_index_id}'")
            deploy_operation = index_endpoint.deploy_index(
                index=index,
                deployed_index_id=deployed_index_id,
                machine_type="e2-highmem-16",
                min_replica_count=1,
                max_replica_count=1,
            )
            print(f"Background task: Index deployment operation name: {deploy_operation.operation.name}")

            # Wait for index deployment to complete
            print("Background task: Waiting for index deployment to complete...")
            deploy_operation.wait()
            print("Background task: Index deployment completed.")


        print(f"Background task: Index '{index_display_name}' created and deployed to endpoint '{endpoint_display_name}'.")

    except Exception as e:
        print(f"Background task: Error during automated index creation and deployment: {e}")

@router.post("/create_index/")
async def create_index(
    teacher: str,
    grade: str,
    subject: str,
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    dimensions: int = 1408,
    distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
    contents_delta_uri: str = None,
    description: str = None,
):
    """Initiates the creation and deployment of a Vertex AI Vector Search index as a background task."""
    print("Received request to create and deploy index.")

    teacher = teacher.replace(' ', '_').lower()
    endpoint_display_name = f"{teacher}_{grade}_{subject}_index"

    # Add the task to the background
    background_tasks.add_task(
        _create_and_deploy_index_task,
        teacher,
        grade,
        subject,
        dimensions,
        distance_measure_type,
        contents_delta_uri,
        description,
        endpoint_display_name
    )

    return {"message": "Index creation and deployment initiated as a background task. Use /check_index_endpoint_status/ to check the status."}

@router.get("/check_index_status/")
async def check_index_status(teacher: str, grade: str, subject: str):
    """Checks the status of a Vertex AI Vector Search index based on teacher, grade, and subject."""
    try:
        # Construct the expected display name
        teacher = teacher.replace(' ', '_').lower()
        display_name = f"{teacher}_{grade}_{subject}_index"

        # List indexes and find the one with the matching display name
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{display_name}"')

        if not indexes:
            return {"status": False, "message": "Index is not yet createt."}


    except Exception as e:
        print(f"Error checking index status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking index status: {e}")


    """Checks the status of the Vertex AI Vector Search Index Endpoint associated with a teacher, grade, and subject."""
    try:
        # Construct the expected index display name to find the associated endpoint
        teacher = teacher.replace(' ', '_').lower()
        index_display_name = f"{teacher}_{grade}_{subject}_index"

        # Find the index first to get the associated endpoint name
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')

        if not indexes:
             raise HTTPException(status_code=404, detail=f"No deployed endpoint found for index '{index_display_name}'. Index may not be deployed yet.")

        index = indexes[0]

        if len(index.deployed_indexes) == 0:
            return {"status": False, "message": "Index Endpoint is not yet Deployed."}

    except Exception as e:
        print(f"Error checking index endpoint status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking index endpoint status: {e}")
    
    return {"status": True, "message": "Index is ready to use."}
    