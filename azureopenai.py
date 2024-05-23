from langchain_openai import AzureChatOpenAI
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from langchain_openai import AzureOpenAIEmbeddings

from config import set_environment

set_environment()

# This requires 'az login' to be run in the terminal
credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

def get_azureopenai_llm(deployment_name: str = "chat"):

    llm = AzureChatOpenAI(
        deployment_name = deployment_name,
        azure_ad_token_provider = token_provider
    )
    return llm

def get_azureopenai_embeddings(model: str = "embedding"):

    return AzureOpenAIEmbeddings(
        azure_ad_token_provider = token_provider,
        model = model
    )