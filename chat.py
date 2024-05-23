import sys
import logging
import pathlib

from typing import Any
from prompt_toolkit import prompt, PromptSession

from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.schema import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from azureopenai import get_azureopenai_llm, get_azureopenai_embeddings

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")

# Load all documents in a directory
def load_documents(directory: str) -> list[Document]:
    directory = pathlib.Path(directory)
    documents = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            documents.extend(load_document(file_path))
    return documents

def load_document(file_path: str | pathlib.Path) -> list[Document]:
    file_path = pathlib.Path(file_path)
    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_path.suffix == ".txt":
        loader = TextLoader(file_path)
    elif file_path.suffix == ".epub":
        loader = EpubReader(file_path)
    elif file_path.suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        logging.warning(f"Unsupported file type: {file_path.suffix}")
        return []
    return loader.load()

def create_vector_db(docs: list[Document], db_path: str):
    embeddings = get_azureopenai_embeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(db_path)

def chat(cache_dir: str):
    embeddings = get_azureopenai_embeddings()
    db = FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
    llm = get_azureopenai_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        Your goal is to answer each question, using the following documents as context, as truthfully as you can.
        If you cannot answer the question or find relevant meaning in the presented texts, tell the user to try re-phrasing the question.

        Put the source document(s) name(s) and page number(s) at the end of the answer
        on a separate line and formatted as "Sources:\n".

        CONTEXT:
        {context}
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_type = "mmr"),
            memory=memory,
            combine_docs_chain_kwargs={ "prompt": ChatPromptTemplate.from_messages( [ system_message_prompt, human_message_prompt, ]), },
        )

    session = PromptSession()

    while True:
        try:
            query = session.prompt("? ")
            if query in ["exit", "quit", "bye"]:
                break
            response = conversation_chain.invoke({"question": query})
            print("\n" + response["answer"] + "\n")
        except (KeyboardInterrupt, EOFError):
            break
        # BUG: sometimes I pass too many tokens to the model when asked about unknown things, why?
        except Exception as e:
            print("I don't know anything about that.")
    print("Goodbye!")

def main():
    cache_dir = "faiss_cache"

    if len(sys.argv) == 1:
        chat(cache_dir)
    elif len(sys.argv) == 2:
        dir = sys.argv[1]
        docs = load_documents(dir)
        create_vector_db(docs, cache_dir)
    else:
        print("Usage: python chat.py [<directory>]")
        sys.exit(1)

if __name__ == "__main__":
    main()