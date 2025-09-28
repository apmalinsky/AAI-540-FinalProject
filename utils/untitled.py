import boto3
import time
import sagemaker
from botocore.exceptions import ClientError
from datetime import datetime

# Configuration
REGION = boto3.Session().region_name
sts_client = boto3.client('sts')
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
 
# The IAM Role Name shared by the team
LAB_ROLE_NAME = 'LabRole' 
# The database created by Feature Store
DB_NAME = 'sagemaker_featurestore' 
# The full name of your Feature Group (Used for verification messages)
FEATURE_GROUP_NAME = 'foodlens-products-feature-group-28-21-51-05' 

# --- 2. Initialize Lake Formation Client ---
lf_client = boto3.client('lakeformation', region_name=REGION)

# The ARN of the specific Principal we are granting access to (e.g., arn:aws:iam::[Your Account]:role/LabRole)
LAB_ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/{LAB_ROLE_NAME}' 

print(f"Preparing to grant SELECT permissions on Feature Store data to IAM Role: {LAB_ROLE_ARN}")

# --- 3. Grant Permissions to the Database and Tables ---
try:
    # 3a. Grant DESCRIBE permission on the Database (Required for Glue/Athena to see the tables)
    lf_client.grant_permissions(
        Principal={'DataLakePrincipalIdentifier': LAB_ROLE_ARN},
        Resource={
            'Database': {'Name': DB_NAME}
        },
        Permissions=['DESCRIBE']
    )

    # 3b. Grant SELECT permission on ALL Tables starting with the FG name prefix within the Database
    # This uses a wildcard to ensure access to all current and future Feature Group tables.
    # Note: We use a wildcard on the table name as Feature Store tables often have a timestamp suffix.
    lf_client.grant_permissions(
        Principal={'DataLakePrincipalIdentifier': LAB_ROLE_ARN},
        Resource={
            'TableWithColumns': {
                'DatabaseName': DB_NAME,
                'TableName': 'foodlens_products_feature_group_*', # Covers all current and future FG tables
                'ColumnWildcard': {} # Grant access to all columns
            }
        },
        Permissions=['SELECT']
    )
    
    print("\n✅ SUCCESS: Permissions granted for Batch/SQL Access.")
    print(f"Role '{LAB_ROLE_NAME}' can now query the Offline Store via Athena.")
    
except lf_client.exceptions.InvalidInputException as e:
    print("\n⚠️ WARNING: You may need to register your S3 location with Lake Formation first.")
    print(f"Error: {e}")
except Exception as e:
    print(f"\n❌ FATAL ERROR: Failed to grant permissions. Ensure your current role has Lake Formation Admin rights. Error: {e}")
