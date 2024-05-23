from langchain_community.llms import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

def get_ollama_llm(model: str = "phi3"):
    return ChatOllama(model=model)

def get_ollama_embeddings():
    return OllamaEmbeddings()