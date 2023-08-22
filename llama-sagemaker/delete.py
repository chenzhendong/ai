
from sagemaker import Predictor

llm = Predictor(
    endpoint_name="huggingface-pytorch-tgi-inference-2023-08-22-17-12-28-521")

llm.delete_model()
llm.delete_endpoint()
