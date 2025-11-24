from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY

def load_llm(provider="openai"):
    if provider == "openai":
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    elif provider == "groq":
        return ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro")
    else:
        raise ValueError("Invalid LLM provider")
