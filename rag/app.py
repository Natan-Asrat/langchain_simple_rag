import streamlit as st
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
import torch


import os
from dotenv import load_dotenv
import sys
package = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="simplerag"


# Caching the tokenizer and model
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

class CustomEmbeddings:
    def embed_documents(self, texts):
        return [get_embedding(chunk.page_content) for chunk in texts]
    
    def embed_query(self, text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Load the persisted Chroma DB
persist_directory = "db_storage"
embedding_model = CustomEmbeddings()
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# Load the document chain
prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {input}
""")

# Create the document chain using ChatGroq
document_chain = create_stuff_documents_chain(ChatGroq(), prompt)

# Create the retriever
retriever = db.as_retriever()

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title("Simple RAG with Groq API on Robotics")

query = st.text_input("Ask a question about robotics:")

if query:
    # Perform the retrieval and get the response
    response = retrieval_chain.invoke({"input": query})
    answer = response['answer']
    
    st.write("### Answer")
    st.write(answer)
