
from sagemaker import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


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


llm = Predictor(
    endpoint_name="huggingface-pytorch-tgi-inference-2023-08-23-01-46-13-498",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer())

# define question and add to messages
instruction = "What are some cool ideas to visit the Washington DC?"
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

