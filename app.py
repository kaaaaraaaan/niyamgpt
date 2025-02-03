import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
from supabase import create_client
import tempfile
from datetime import datetime
from components.sidebar import render_sidebar

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize OpenAI embeddings and chat model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-turbo-preview")

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

# Set page config
st.set_page_config(
    page_title="NiyamGPT",
    page_icon="🤖",
    layout="centered",
)

# Load CSS files
def load_css(css_files):
    css = ""
    for css_file in css_files:
        with open(css_file) as f:
            css += f.read() + "\n"
    return f'<style>{css}</style>'

st.markdown(load_css(['static/style.css', 'static/sidebar.css']), unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "next_chat_id" not in st.session_state:
    st.session_state.next_chat_id = 1
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Create first chat session if none exists
if not st.session_state.chat_sessions:
    first_chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1
    st.session_state.chat_sessions.append({
        "id": first_chat_id,
        "title": f"Session #{first_chat_id}",
        "messages": [],
        "created_at": str(datetime.now())
    })
    st.session_state.current_chat_id = first_chat_id
    st.session_state.messages = []
    st.session_state.chat_history = []

# Title
st.title("NiyamGPT")

# Render sidebar
render_sidebar()

# Display current chat in main area
if st.session_state.current_chat_id:
    current_session = next((s for s in st.session_state.chat_sessions if s['id'] == st.session_state.current_chat_id), None)
    if current_session:
        for message in current_session['messages']:
            with st.container():
                st.markdown(f"""
                <div class="chat-message {message['role']}">
                    <img class="avatar" src="{'https://api.dicebear.com/7.x/bottts/svg?seed=1' if message['role'] == 'assistant' else 'https://api.dicebear.com/7.x/avataaars/svg?seed=2'}" />
                    <div class="message">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)

# Chat input
if st.session_state.current_chat_id and (prompt := st.chat_input("Send a message")):
    current_session = next((s for s in st.session_state.chat_sessions if s['id'] == st.session_state.current_chat_id), None)
    if current_session:
        # Add user message to chat history
        current_session['messages'].append({"role": "user", "content": prompt})
        st.session_state.messages = current_session['messages']
        
        with st.spinner("Thinking..."):
            try:
                # Get response from QA chain
                response = qa_chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                
                # Update chat history
                st.session_state.chat_history.extend([
                    (prompt, answer)
                ])
                
                # Add assistant response to chat history
                assistant_message = {"role": "assistant", "content": answer}
                current_session['messages'].append(assistant_message)
                st.session_state.messages = current_session['messages']
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                current_session['messages'].append({"role": "assistant", "content": error_message})
        
        # Rerun to update the chat display
        st.rerun()
elif not st.session_state.current_chat_id:
    st.info("Click '+ New Chat' to start a conversation")
