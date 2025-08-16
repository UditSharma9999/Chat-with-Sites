import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ====== This is only for Streamlit deployment (SQLite workaround) ============== #
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ============================================================================== #

# Load Hugging Face API Token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_vectorstore_from_url(url):
    """Load website content, split into chunks, and build vector store."""
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(document_chunks, embedding)

    return vector_store


def get_context_retriever_chain(vector_store):
    """Create a retriever chain with history awareness."""
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1-0528",
            task="conversational",   # ✅ use conversational task
            max_new_tokens=512,
            huggingfacehub_api_token=hf_token,
        )
    )
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    

def get_conversational_rag_chain(retriever_chain):
    """Build full RAG chain for conversational Q&A."""
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1-0528",
            task="conversational",   # ✅ use conversational task
            max_new_tokens=512,
            huggingfacehub_api_token=hf_token,
        )
    )
    prompt = ChatPromptTemplate.from_messages([
        # ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        ("system", 
        "You are a helpful assistant. Use ONLY the information provided in {context} to answer. "
        "If the answer is not contained in the context, reply with 'I don't know' in 1-2 lines."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    """Fetch response from conversational RAG chain."""
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


# ==================== Streamlit App ==================== #
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url.strip() == "":
    st.info("Please enter a website URL")

else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


