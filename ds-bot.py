import numpy as np
import streamlit as st
import PyPDF2, docx2txt, faiss
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
)
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

st.title("Deep Seek Model Chatbot")

# Initialize the session state
if 'document' not in st.session_state:
    st.session_state['documents'] = []
    st.session_state['vector_store'] = None

with st.sidebar:
    st.header("Select your Deep Seek Model:")
    select_model = st.selectbox(
        "Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b"], index=0
    )
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    st.markdown("Built with [Ollama](https://ollama.com/) | [Langchain](https://python.langchain.com/docs/introduction/)")

# Initialize chat engine
llm_engine = ChatOllama(
    model=select_model,
    base_url="http://localhost:11434",
    temperature=0.3)

# Function to extract info from PDF files
def extract_text_from_pdf(pdf_file):
    text = ''
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

# Function to extract info from DOCX files
def extract_text_from_docs(docs_file):
    return docx2txt.process(docs_file)

# Load and process document
if uploaded_file:
    text_data = ''
    if uploaded_file.type == "application/pdf":
        text_data += extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text_data += extract_text_from_docs(uploaded_file)
    else:
        text_data += uploaded_file.read().decode("utf-8")

    if text_data:
        st.session_state['documents'].append(text_data)
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_chunks = text_data.split(". ")  # Splitting into sentences
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        st.session_state['vector_store'] = FAISS.from_documents(documents, embed_model)
        st.success("Document processed successfully!")

# System message prompt template
prompt_template = SystemMessagePromptTemplate.from_template(
    "You are an AI expert who can read and understand documents. "
    "You will provide answers to questions related to the documents provided to you."
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "BOT", "content": "Hi, I am your bot. Upload a document and ask me questions about it!"}
    ]

# Chat display
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Type your question here...")

def generate_ai_response(prompt_chain):
    pipeline = prompt_chain | llm_engine | StrOutputParser()
    return pipeline.invoke({})

def retrieve_relevant_text(query):
    if "vector_store" in st.session_state:
        docs = st.session_state['vector_store'].similarity_search(query, k=1)
        return "\n".join([doc.page_content for doc in docs])
    return ""

def build_prompt_chain():
    prompt_sequence = [prompt_template]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "BOT":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    if "vector_store" in st.session_state:
        relevant_text = retrieve_relevant_text(user_query)
        if relevant_text:
            st.session_state.message_log.append({"role": "user", "content": user_query})
            with st.spinner("Processing..."):
                prompt_chain = build_prompt_chain()
                ai_response = generate_ai_response(prompt_chain)
            st.session_state.message_log.append({"role": "BOT", "content": ai_response})
        else:
            st.session_state.message_log.append({"role": "BOT", "content": "No relevant information found in the uploaded document."})
    else:
        st.session_state.message_log.append({"role": "BOT", "content": "Please upload a document first before asking questions."})
    st.rerun()
