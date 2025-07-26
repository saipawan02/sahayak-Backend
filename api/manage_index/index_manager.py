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

def create_index_endpoint(display_name: str):
    """Creates a Vertex AI Vector Search Index Endpoint."""
    try:
        index_endpoint = aiplatform.IndexEndpoint.create(
            display_name=display_name,
            public_endpoint_enabled=True,  # Set to True to get a public endpoint
        )
        print(f"Index Endpoint creation operation initiated: {index_endpoint.operation.name}")
        return index_endpoint
    except Exception as e:
        print(f"Error creating Index Endpoint: {e}")
        raise e

def deploy_index_to_endpoint(index: aiplatform.MatchingEngineIndex, index_endpoint: aiplatform.MatchingEngineIndexEndpoint, deployed_index_id: str):
    """Deploys a Vertex AI Vector Search Index to an Index Endpoint."""
    try:
        operation = index_endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id,
            machine_type="e2-highmem-16",  # Adjust machine type as needed
            min_replica_count=1,
            max_replica_count=1,
        )
        print(f"Index deployment operation initiated: {operation.operation.name}")
        return operation
    except Exception as e:
        print(f"Error deploying index to endpoint: {e}")
        raise e

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
        # 1. Create the index
        index_display_name = f"{teacher}_{grade}_{subject}_index"
        print(f"Background task: Initiating creation of index: {index_display_name}")
        index_operation = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=endpoint_display_name,
            contents_delta_uri=contents_delta_uri,
            description="Matching Engine Index",
            dimensions=100,
            approximate_neighbors_count=150,
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7,
            index_update_method="BATCH_UPDATE",  # Options: STREAM_UPDATE, BATCH_UPDATE
            distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        )
        print(f"Background task: Index creation operation name: {index_operation.operation.name}")

        # Wait for index creation to complete
        print("Background task: Waiting for index creation to complete...")
        index_operation.wait_for_resource()
        index = index_operation.result()
        print(f"Background task: Index created: {index.resource_name}")

        # 2. Find or create the index endpoint
        print(f"Background task: Finding or creating index endpoint: {endpoint_display_name}")
        endpoints = aiplatform.IndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
        if endpoints:
            index_endpoint = endpoints[0]
            print(f"Background task: Found existing index endpoint: {index_endpoint.resource_name}")
        else:
            print(f"Background task: Creating new index endpoint: {endpoint_display_name}")
            endpoint_operation = aiplatform.IndexEndpoint.create(
                display_name=endpoint_display_name,
                public_endpoint_enabled=True,
            )
            print(f"Background task: Index endpoint creation operation name: {endpoint_operation.operation.name}")
            endpoint_operation.wait()
            index_endpoint = endpoint_operation.result()
            print(f"Background task: Index endpoint created: {index_endpoint.resource_name}")


        # 3. Deploy the index to the endpoint
        deployed_index_id = f"{teacher}_{grade}_{subject}_deployed_index" # Unique ID for the deployed index
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
        # In a real application, you might want to log this error or update a status in a database

@router.post("/create_index/")
async def create_index(
    teacher: str,
    grade: str,
    subject: str,
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    dimensions: int = 1408,
    distance_measure_type: str = "DOT_PRODUCT_DENORMALIZE",
    contents_delta_uri: str = None,
    description: str = None,
):
    """Initiates the creation and deployment of a Vertex AI Vector Search index as a background task."""
    print("Received request to create and deploy index.")

    teacher = teacher.replace(' ', '_')
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
        display_name = f"{teacher}_{grade}_{subject}_index"

        # List indexes and find the one with the matching display name
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{display_name}"')

        if not indexes:
            raise HTTPException(status_code=404, detail=f"Index with display name '{display_name}' not found.")

        # Assuming there's only one index with this display name
        index = indexes[0]

        # Get the latest operation for the index
        latest_operation = index.latest_future

        status = {
            "name": index.resource_name,
            "display_name": index.display_name,
            "etag": index.etag,
            "create_time": index.create_time.isoformat() if index.create_time else None,
            "update_time": index.update_time.isoformat() if index.update_time else None,
            "index_datastream": index.index_datastream.to_dict() if index.index_datastream else None,
            "metadata": index.metadata.to_dict() if index.metadata else None,
            "deployed_indexes": [deployed_index.to_dict() for deployed_index in index.deployed_indexes],
            "latest_operation_name": latest_operation.operation.name if latest_operation else None,
            "latest_operation_done": latest_operation.operation.done if latest_operation else None,
            "latest_operation_state": latest_operation.operation.metadata.partial_dict().get('generic_metadata', {}).get('state', 'UNKNOWN') if latest_operation else None,
        }

        if latest_operation and latest_operation.operation.error:
            status["latest_operation_error"] = latest_operation.operation.error.message

        return {"status": status}

    except Exception as e:
        print(f"Error checking index status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking index status: {e}")


@router.get("/check_index_endpoint_status/")
async def check_index_endpoint_status(teacher: str, grade: str, subject: str):
    """Checks the status of the Vertex AI Vector Search Index Endpoint associated with a teacher, grade, and subject."""
    try:
        # Construct the expected index display name to find the associated endpoint
        teacher = teacher.replace(' ', '_')
        index_display_name = f"{teacher}_{grade}_{subject}_index"

        # Find the index first to get the associated endpoint name
        indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')

        if not indexes:
             raise HTTPException(status_code=404, detail=f"Index with display name '{index_display_name}' not found. Index or its deployment may not have been initiated yet.")

        index = indexes[0]

        if not index.deployed_indexes:
             raise HTTPException(status_code=404, detail=f"No deployed endpoint found for index '{index_display_name}'. Index may not be deployed yet.")

        # Assuming the first deployed index gives us the endpoint
        deployed_index = index.deployed_indexes[0]
        endpoint_resource_name = deployed_index.index_endpoint # This is the full resource name of the endpoint

        # Get the Index Endpoint object
        index_endpoint = aiplatform.IndexEndpoint(endpoint_resource_name)

        status = {
            "name": index_endpoint.resource_name,
            "display_name": index_endpoint.display_name,
            "etag": index_endpoint.etag,
            "create_time": index_endpoint.create_time.isoformat() if index_endpoint.create_time else None,
            "update_time": index_endpoint.update_time.isoformat() if index_endpoint.update_time else None,
            "public_endpoint_domain_name": index_endpoint.public_endpoint_domain_name,
            "network": index_endpoint.network, # VPC network if applicable
            "enable_private_service_connect": index_endpoint.enable_private_service_connect,
            "deployed_indexes": [dep_index.to_dict() for dep_index in index_endpoint.deployed_indexes],
            # You can add more endpoint-specific details here
        }

        return {"status": status}

    except Exception as e:
        print(f"Error checking index endpoint status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking index endpoint status: {e}")