import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from mlflow.deployments import get_deploy_client

from utils.utils import load_config_file

app_name = load_config_file().get('app_name')
arn = load_config_file().get('arn') # Função - Role
image_ecr_uri = load_config_file().get('image_ecr_uri')
region = load_config_file().get('region')

model_uri = load_config_file().get('model_uri')

config = dict(
                execution_role_arn=arn,
                bucket_name="New-s3-bucket",
                image_url=image_ecr_uri,
                region_name=region,
                archive=False,
                instance_type="ml.m4.xlarge",
                instance_count=1,
                synchronous=True,
                timeout_seconds=3600,
                variant_name="prod-variant-3",
                tags={"training_timestamp": "2023-12-05"},
            )

client = get_deploy_client("sagemaker")
deploy_client = client.create_deployment(app_name,
                                         model_uri=model_uri,
                                         flavor='python_function',
                                         config=config)
print(f"Deploy_client: {deploy_client}")