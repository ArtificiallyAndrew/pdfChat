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
from PIL import Image


st.title("Chat with your PDF 💬")
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
    st.write("Upload a PDF file:")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    query = st.text_input("Enter a question:", "")
    
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
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3,streaming=True),verbose=True,retriever=vectorstore.as_retriever(),memory=memory)
        
        if query: # if there's a user query
              history = [] # Initialize chat history, you can modify this based on your needs
              with get_openai_callback() as cost:
                response = generate_response(chain, history, query)
                st.write(f"Response: {response}")
                print(cost)                                   
                    
if __name__ == "__main__":
    main()
           