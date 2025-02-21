import streamlit as st
import os, tempfile, docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

PROMPT_TEMPLATE = """
You are an AI assistant specialized in analyzing and answering questions based on uploaded documents.  

### Instructions:
1. Carefully analyze the provided document context and the user’s query.
2. Your response should be **highly relevant** to the document content.
3. **Do not generate answers** based on external knowledge—use only the given context.
4. If the user asks for the title, look for the most prominent heading in the document.
5. If the user asks for a summary, focus on the key points in the document.
6. If the document does not contain enough relevant information to answer the query, respond with:  
   **"This question is not aligned with the context of the uploaded document."**  
7. Maintain a professional, and informative tone in your response.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Streamlit Sidebar
with st.sidebar:
    model_selection = st.sidebar.selectbox("Select DeepSeek Model", ["deepseek-r1:1.5b", "deepseek-r1:7b"], index=0)
    uploaded_file = st.file_uploader("Upload your document (PDF or DOCX)", type=["pdf", "docx"], help="Select a PDF or DOCX document for analysis")
    st.markdown('Built with [DeepSeek](https://deepseek.com/) | [Langchain](https://python.langchain.com/docs/introduction/)')

# Define Embedding Model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH = "faiss_rag_index"

# Function to Save Uploaded File
def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Function to Load Documents
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
        return [Document(page_content=text)]
    else:
        raise ValueError("Unsupported file format")

# Function to Chunk Documents
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True)
    return text_processor.split_documents(raw_documents)

# Function to Index Documents
def index_documents(document_chunks):
    faiss_index = FAISS.from_documents(document_chunks, EMBEDDING_MODEL)
    faiss_index.save_local(VECTOR_DB_PATH)

# Function to Retrieve Relevant Documents
def find_related_documents(query):
    faiss_index = FAISS.load_local(VECTOR_DB_PATH, EMBEDDING_MODEL,allow_dangerous_deserialization=True)
    return faiss_index.similarity_search(query, k=5) 

# Function to Generate AI Response
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = Ollama(model=model_selection)
    response_chain = prompt | llm
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Streamlit UI
st.header("Welcome Human :)")
st.subheader("How may I assist you with the document today?")

if uploaded_file:
    user_input = st.chat_input("Enter your question about the document...")
    
    with st.spinner("Processing document... ⏳"):
        saved_path = save_uploaded_file(uploaded_file)
        raw_docs = load_document(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
    st.success("✅ Document processed and indexed successfully!")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document with RAG... ⏳"):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
