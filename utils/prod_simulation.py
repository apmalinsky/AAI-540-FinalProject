import json
import time
import uuid
from datetime import datetime
from time import sleep

import pandas as pd
from sagemaker.s3 import S3Uploader

# Helper function for prod data simulation
def upload_ground_truth(records, upload_path, timestamp):
    # Convert records to JSON-lines format
    ground_truth_data_to_upload = "\n".join([json.dumps(r) for r in records])
    
    # Set S3 path based on timestamp
    target_s3_uri = f"{upload_path}/{timestamp:%Y/%m/%d/%H}/ground_truth_{timestamp:%M%S}.jsonl"
    
    print(f"  Uploading {len(records)} ground truth records to {target_s3_uri}")
    S3Uploader.upload_string_as_file_body(ground_truth_data_to_upload, target_s3_uri)

# Define live traffic simulation
def simulate_live_traffic_for_duration(
    endpoint_name,
    sagemaker_session,
    prod_df,
    gt_upload_path,
    duration_hours,
    sample_size=500,        
    wait_time_seconds=600 
):
    print(f"--- Starting Traffic Simulation for {duration_hours} hours ---")
    print(f"Sending {sample_size} records every {wait_time_seconds / 60} minutes.")
    
    # Get the start time and calculate the total duration in seconds
    start_time = time.time()
    duration_seconds = duration_hours * 3600

    # Loop until kernel restart or reached max duration
    while time.time() - start_time < duration_seconds:
        try:
            print(f"\n--- Generating new data batch ---")
            
            sample_traffic_df = prod_df.sample(n=sample_size)
            ground_truth_records_for_this_batch = []
            current_timestamp = datetime.utcnow()

            for index, row in sample_traffic_df.iterrows():
                 # First column is the label (true score) and the rest are features
                true_label = row.iloc[0]
                features = row.iloc[1:]
                features_payload = ",".join(features.astype(str).values) + "\n"
                inference_id = str(uuid.uuid4()) # generate unique id
                sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType="text/csv",
                    Body=features_payload,
                    InferenceId=inference_id,
                )
                ground_truth_records_for_this_batch.append({
                    "groundTruthData": {"data": str(true_label), "encoding": "CSV"},
                    "eventMetadata": {"eventId": inference_id},
                    "eventVersion": "0",
                })
            
            print(f"Sent {sample_size} predictions to endpoint.")

            # Wait 5 minutes to give the SageMaker Data Capture service
            # time to write its files to the correct S3 hourly folder.
            print("Waiting 5 minutes for data capture to land in S3...")
            sleep(300) 
            
            upload_ground_truth(
                ground_truth_records_for_this_batch,
                gt_upload_path,
                current_timestamp
            )
            
            # Check if the next wait period would exceed the duration
            if (time.time() - start_time + wait_time_seconds) > duration_seconds:
                print("\nDuration reached. Stopping simulation after this batch.")
                break

            print(f"Batch complete. Waiting {wait_time_seconds / 60} minutes...")
            sleep(wait_time_seconds)

        except Exception as e:
            print(f"Error in simulation loop: {e}")
            print("Restarting loop after a 1-minute wait...")
            sleep(60)
            
    print(f"\n--- Simulation finished after approximately {duration_hours} hours. ---")