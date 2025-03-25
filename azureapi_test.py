from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
load_dotenv()

client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["DEEPSEEKR1_GITHUB_TOKEN"]),
)

response = client.complete(
    messages=[
        UserMessage("Hello, how are you?"),
    ],
    model="DeepSeek-R1",
    max_tokens=50,
)

print(response)