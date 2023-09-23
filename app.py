from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import openai
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import streamlit as st
import os
import tiktoken
from PIL import Image


st.title("CFA PDF Chatbot")
def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_response(chain, history, query):
    result = chain(
        {"question": query, 'chat_history': history}, return_only_outputs=True)
    return result["answer"]

def main():
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    st.write("A chat bot with PDF analysis skills can help you find important information in PDF files by answering questions you ask it. ")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    query = st.text_input("Enter a question or ask to summarize:", "")
    
    if pdf_file is not None:
        text = process_pdf(pdf_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        general_system_template = r""" 
         You are a helpful assistant that works for NYU.  
        """
        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            ]
        qa_prompt = ChatPromptTemplate.from_messages( messages )
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo",max_tokens=4000,temperature=0.3,streaming=True),retriever=vectorstore.as_retriever(),memory=memory)
        
        if query: # if there's a user query
              history = [] # Initialize chat history, you can modify this based on your needs
              with get_openai_callback() as cost:
                response = generate_response(chain, history, query)
                st.write(f"Response: {response}")
                print(cost)                                   
                    
if __name__ == "__main__":
    main()
           
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            footer: after {
	        content:'iTLAB  -  2023'; 
	        visibility: visible;
	        display: block;
	        position: relative;
	        #background-color: purple;
	        padding: 5px;
	        top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


