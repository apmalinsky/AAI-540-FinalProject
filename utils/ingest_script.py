# This script was created with AI-assistance from Google Gemini
import sys
import boto3
import pandas as pd
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from sagemaker import Session as SagemakerSession
from sagemaker.feature_store.feature_group import FeatureGroup

# Get arguments passed from the notebook job submission
args = getResolvedOptions(
    sys.argv, 
    ['JOB_NAME', 's3_data_path', 'feature_group_name', 'sagemaker_role_arn', 'region']
)

# Initialize Spark and Glue Contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Configure Feature Store Session and Arguments
REGION = args['region']
FEATURE_GROUP_NAME = args['feature_group_name']
RECORD_IDENTIFIER_NAME = 'code' 

# Read the processed data from S3
print(f"Reading data from {args['s3_data_path']}...")
df_spark = spark.read.parquet(args['s3_data_path'])
total_records = df_spark.count()

# Implement ingestion using per-record ingestion on partitions
print(f"Ingesting {total_records} records into {FEATURE_GROUP_NAME} using parallel per-record writes...") 

REQUIRED_FEATURE_ORDER = [
  'code', 'product_name', 'nova_group', 'additives_n', 'ingredients_n', 
  'nutriscore_score', 'nova_group_100g', 'energy_100g', 'sodium_100g', 
  'proteins_100g', 'fruits_vegetables_legumes_estimate_from_ingredients_100g', 
  'salt_100g', 'nutrition_score_fr_100g', 'carbohydrates_100g', 
  'energy_kcal_100g', 'fruits_vegetables_nuts_estimate_from_ingredients_100g',
  'sugars_100g', 'fat_100g', 'saturated_fat_100g', 'fiber_100g', 
  'trans_fat_100g', 'vitamin_a_100g', 'cholesterol_100g', 'calcium_100g', 
  'iron_100g', 'vitamin_c_100g', 'EventTime'
]

# Select and reorder the columns in the Spark DataFrame
df_spark = df_spark.select(*REQUIRED_FEATURE_ORDER)

# Repartitioning ensures a manageable number of smaller batches.
NUM_BATCHES = 10 
df_spark = df_spark.repartition(NUM_BATCHES)

def ingest_batch(iterator):
    """
    Function executed on each Spark partition/batch.
    """
    # Re-initialize necessary Boto3/SageMaker objects for this partition
    boto_session = boto3.Session(region_name=REGION)
    featurestore_runtime = boto_session.client('sagemaker-featurestore-runtime', region_name=REGION)

    for record_dict in iterator:
        # PySpark Row objects need to be converted to dictionary for field access
        record_dict = record_dict.asDict()

        # Prepare the list of feature name/value pairs
        feature_list = []
        
        # Build the Feature list using the correct API structure
        for k, v in record_dict.items():
            feature_list.append({
                'FeatureName': k, 
                'ValueAsString': str(v) if v is not None else ''
            })
        
        if record_dict.get(RECORD_IDENTIFIER_NAME) and record_dict.get('EventTime'):
            try:
                featurestore_runtime.put_record(
                    FeatureGroupName=FEATURE_GROUP_NAME,
                    Record=feature_list 
                )
            except Exception as e:
                # Log individual record ingestion failure but continue the job
                print(f"Error ingesting record {record_dict.get(RECORD_IDENTIFIER_NAME)}: {e}")
        else:
            print(f"Skipping record due to missing code or EventTime: {record_dict}")

# Run the ingestion across all Spark partitions/batches
df_spark.foreachPartition(ingest_batch)

print("Ingestion complete via AWS Glue job.")

# Commit the job for Glue to log success
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()
