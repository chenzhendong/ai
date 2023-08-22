import sagemaker
import boto3
import json
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel
from dotenv import load_dotenv

load_dotenv()
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")


# sagemaker config
instance_type = "ml.g5.2xlarge"
number_of_gpu = 1
health_check_timeout = 300


# Define Model and Endpoint configuration parameter
config = {
  'HF_MODEL_ID': "meta-llama/Llama-2-7b-chat-hf", # model_id from hf.co/models
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(2048),  # Max length of the generation (including input text)
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(4096),  # Limits the number of tokens that can be processed in parallel during the generation
  'HUGGING_FACE_HUB_TOKEN': os.getenv('HUGGING_FACE_HUB_TOKEN')
  # ,'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
}

# check if token is set
# assert config['HUGGING_FACE_HUB_TOKEN'] != "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", "Please set your Hugging Face Hub token"

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  env=config
)


# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
  initial_instance_count=1,
  instance_type=instance_type,
  # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
)


# print("Endpoint name:", llm)

# # Set up boto3 client for SSM
# ssm_client = boto3.client('ssm')

# # Define the Parameter Store key
# parameter_key = "/com/commoninf/zchen/sagemaker/endpoint1"

# # Save the endpoint value to Parameter Store
# ssm_client.put_parameter(
#     Name=parameter_key,
#     Value=llm,
#     Type='String',
#     Overwrite=True
# )
