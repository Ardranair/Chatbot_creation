import streamlit as st
from models.llm import load_llm
from utils.rag_utils import build_vectorstore, retrieve

st.title("🤖 AI RAG Chatbot")

llm_choice = st.sidebar.selectbox("LLM Provider", ["openai", "groq", "gemini"])
response_mode = st.sidebar.radio("Response Style", ["Concise", "Detailed"])
build_clicked = st.sidebar.button("📌 Build Knowledge Base")

if build_clicked:
    with st.spinner("Building knowledge base..."):
        build_vectorstore()
    st.success("Knowledge base ready!")

query = st.text_input("Ask something...")

if st.button("Send") and query:
    llm = load_llm(llm_choice)
    results = retrieve(query)

    context = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    Use the following context to answer.

    Context:
    {context}

    Question: {query}

    Answer style: {response_mode}
    """

    response = llm.invoke(prompt)
    st.write("### Response:")
    st.write(response.content)
