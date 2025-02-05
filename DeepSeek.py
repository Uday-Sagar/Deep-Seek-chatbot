import os
import PyPDF2
import docx2txt
import streamlit as st
import numpy as np
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
# from llama_index.llms import LangchainLLM

#initialize the session state
if 'document' not in st.session_state:
    st.session_state['documents']=[]
    st.session_state['faiss_index'] = None

st.title('Document chatbot')
st.write('Upload your documents and ask questions about your document')

with st.sidebar:
    # Document upload in the sidebar
    files_upload=st.file_uploader('Upload your documents here:',type=['pdf','docx'],accept_multiple_files=True)
    st.markdown('Built with [Ollama](https://ollama.com/) | [Langchain](https://python.langchain.com/docs/introduction/)')


#function to extract info from the PDF files
def extract_text_from_pdf(pdf_file):
    text=''
    reader=PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text+=page.extract_text()+ '\n'
    return text

#function to extract info from docx files
def extract_text_from_docs(docs_file):
    return docx2txt.process(docs_file)

if files_upload:
    text_data=''
    for file in files_upload:
        if file.type == "application/pdf":
            text_data += extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_data += extract_text_from_docs(file)
    if text_data:
        with open("documents/temp.txt", "w", encoding="utf-8") as f:
            f.write(text_data)
        st.session_state['documents'].append("documents/temp.txt")
        st.success("Documents processed successfully!")

#initialize the embeddings
if st.session_state['documents'] and not st.session_state['faiss_index']:
    text_data = ""
    for doc in st.session_state['documents']:
        with open(doc, "r", encoding="utf-8") as f:
            text_data += f.read()

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.embed_documents([text_data])
    embeddings = np.array(embeddings).astype(np.float32)
    dimension = embeddings.shape[1]  # dimension of the embeddings
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    st.session_state['faiss_index'] = faiss_index

#Building the face of the app
query=st.text_input('Ask your question about your documents')
if query and st.session_state['faiss_index']:
    query_embedding = embed_model.embed_query(query)
    query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state['faiss_index'].search(query_embedding, k=1)
    if indices.size > 0:
        doc_index = indices[0][0]
        doc_text = text_data
    
    llm = OllamaLLM(model="deepseek-r1:1.5b")
    response = llm.predict(f"Context: {doc_text}\n\nUser Query: {query}")
    
    st.write("### Answer:")
    st.write(response)