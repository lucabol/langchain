from langchain_community.llms import Ollama
llm = Ollama(model="phi3")
result = llm.invoke("how can langsmith help with testing?")
print(result)
