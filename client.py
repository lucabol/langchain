from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")
response = remote_chain.invoke({
    "input": "weather in javea, spain?",
    "chat_history": []  # Providing an empty list as this is the first call
})
print(response["output"])