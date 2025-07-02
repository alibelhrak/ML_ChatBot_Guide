from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st


st.title("Machine Learning Chatbot Guide")

loader = PyPDFLoader(r'C:\Users\alibe\Downloads\ML_Chatbot_Guide\ML_For_Beginners.pdf')
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    groq_api_key=Grok_API_KEY,
    model_name="llama3-70b-8192"    
  
)
query = st.text_input("Ask a question about Machine Learning:")
prompt = query

system_prompt = (
    "You are a helpful assistant for question answering tasks.\n"
    "Use the following pieces of context to answer the question.\n"
    "If you don't know the answer, say 'I don't know'.\n"
    "Don't make up an answer.\n\n"
    "Context:\n{context}"
) 
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])  

question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  
if query:
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])