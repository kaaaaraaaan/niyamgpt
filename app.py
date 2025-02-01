import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
from supabase import create_client
import tempfile

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize OpenAI embeddings and chat model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Initialize vector store
vector_store = SupabaseVectorStore(
    supabase,
    embeddings,
    table_name="document_chunks",
    query_name="match_documents"
)

# Initialize conversation chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

# Set page config to make it look like ChatGPT
st.set_page_config(
    page_title="NiyamGPT",
    page_icon="🤖",
    layout="centered",
)

# Load external CSS
def load_css(css_file):
    with open(css_file) as f:
        return f'<style>{f.read()}</style>'

st.markdown(load_css('static/style.css'), unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("NiyamGPT")

# Display current chat in main area
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <img class="avatar" src="{'https://api.dicebear.com/7.x/bottts/svg?seed=1' if message['role'] == 'assistant' else 'https://api.dicebear.com/7.x/avataaars/svg?seed=2'}" />
            <div class="message">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# Sidebar with chat history
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2ZM20 16H5.17L4 17.17V4H20V16Z" fill="currentColor"/>
            <path d="M7 9H17M7 13H14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Chat History</span>
    </div>
    """, unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <img class="avatar" src="{'https://api.dicebear.com/7.x/bottts/svg?seed=1' if message['role'] == 'assistant' else 'https://api.dicebear.com/7.x/avataaars/svg?seed=2'}" />
                <div class="message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Send a message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Get response from QA chain
            response = qa_chain({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            source_docs = response.get("source_documents", [])
            
            # Debug: Print retrieved documents
            st.write("Debug - Retrieved Documents:")
            for i, doc in enumerate(source_docs):
                st.write(f"Document {i+1}:")
                st.write(f"Content: {doc.page_content[:200]}...")
                st.write(f"Metadata: {doc.metadata}")
            
            # Update chat history
            st.session_state.chat_history.extend([
                (prompt, answer)
            ])
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Rerun to update the chat display
    st.rerun()
