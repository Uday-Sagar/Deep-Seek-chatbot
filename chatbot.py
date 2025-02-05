import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
)

st.title('Deep Seek Model')
with st.sidebar:
    st.header('Select your Deep Seek Model:')
    select_model=st.selectbox(
        'Choose Model',['deepseek-r1:1.5b','deepseek-r1:7b'],index=0
    )
    st.markdown('Build with [Ollama](https://ollama.com/) | [Langchain](https://python.langchain.com/docs/introduction/)')


#initiate the chatengine
llm_engine=ChatOllama(
    model=select_model,
    base_url='http://localhost:11434',
    temperature=0.3
)

#system message prompt tempalte
prompt_template=SystemMessagePromptTemplate.from_template('You are an AI expert who can read documents and can understand them.'
                                'You will provide answers to the questions that are asked regarding the documents give to you')

if 'message_log' not in st.session_state:
    st.session_state.message_log=[
        {'role':'BOT',
         'content':'Hi I am your bot, created to answer your questions about the documents'}]
    
#create the chat container
chat_container=st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

#chat input
user_query=st.chat_input('Type your question here...') 

def generate_ai_response(prompt_chain):
    pipeline=prompt_chain | llm_engine | StrOutputParser()
    return pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence=[prompt_template]
    for msg in st.session_state.message_log:
        if msg['role']=='user':
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg['role']=='BOT':
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "BOT", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
