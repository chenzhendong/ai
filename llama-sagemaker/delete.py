
import boto3, os
from dotenv import load_dotenv
from sagemaker.huggingface.model import HuggingFacePredictor


load_dotenv()
ssm_client = boto3.client('ssm')

# Define the Parameter Store key
parameter_key = os.getenv('ENDPOINT_KEY')

# Save the endpoint value to Parameter Store
result = ssm_client.get_parameter(
    Name=parameter_key,
    WithDecryption=False
)

endpoint_name = result['Parameter']['Value']


llm = HuggingFacePredictor(
    endpoint_name=endpoint_name)

llm.delete_model()
llm.delete_endpoint()

print("Model and endpoint are deleted.")
