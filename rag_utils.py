import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from models.embeddings import load_embeddings

def load_documents(directory="project/data/"):
    docs = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, file))
            docs.extend(loader.load())
    return docs

def build_vectorstore():
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    return vectorstore

def retrieve(query):
    vectorstore = Chroma(persist_directory="db", embedding_function=load_embeddings())
    return vectorstore.similarity_search(query, k=3)
