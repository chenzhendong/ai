
import boto3, os
from dotenv import load_dotenv
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.huggingface.model import HuggingFacePredictor


def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt

messages = [
  { "role": "system","content": "You are a friendly and knowledgeable vacation planning assistant named Clara. Your goal is to have natural conversations with users to help them plan their perfect vacation. "}
]

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
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer())

while True:
  instruction = input("Enter a sentence: ")
  if instruction == "":
      print("Exiting ...")
      break
  else:
      print("You entered:", instruction)

  messages.append({"role": "user", "content": instruction})
  prompt = build_llama2_prompt(messages)

  chat = llm.predict({"inputs":prompt})

  # print(chat[0]["generated_text"][len(prompt):])

  # hyperparameters for llm
  payload = {
    "inputs":  prompt,
    "parameters": {
      "do_sample": True,
      "top_p": 0.6,
      "temperature": 0.9,
      "top_k": 50,
      "max_new_tokens": 512,
      "repetition_penalty": 1.03,
      "stop": ["</s>"]
    }
  }
  # send request to endpoint
  response = llm.predict(payload)
  print(response[0]["generated_text"][len(prompt):])

