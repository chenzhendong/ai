
from sagemaker import Predictor

llm = Predictor(
    endpoint_name="huggingface-pytorch-tgi-inference-2023-08-23-01-46-13-498")

llm.delete_model()
llm.delete_endpoint()
