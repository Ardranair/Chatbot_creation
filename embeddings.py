from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_API_KEY

def load_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
