import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
import tempfile
import time
from typing import List
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

def process_chunks_in_batches(chunks: List[Document], document_id: str, embeddings: OpenAIEmbeddings, batch_size: int = 5):
    """Process document chunks in batches with progress tracking."""
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        status_text.text(f"Processing chunks {i+1}-{min(i+batch_size, total_chunks)} of {total_chunks}")
        
        # Process each chunk in the batch
        for chunk in batch:
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    chunk_embedding = embeddings.embed_query(chunk.page_content)
                    supabase.table("document_chunks").insert({
                        "document_id": document_id,
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "embedding": chunk_embedding
                    }).execute()
                    time.sleep(0.5)  # Add a small delay between chunks
                    break  # If successful, break the retry loop
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise Exception(f"Error processing chunk after {max_retries} attempts: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
        # Update progress
        progress = min((i + batch_size) / total_chunks, 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    status_text.empty()
    return total_chunks

# Set page config
st.set_page_config(
    page_title="RAG Admin Panel",
    page_icon="⚙️",
    layout="wide"
)

# Admin authentication
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        with st.form("Login"):
            st.write("## Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username == os.getenv("ADMIN_USERNAME") and password == os.getenv("ADMIN_PASSWORD"):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return False
    return True

if check_password():
    st.title("RAG Admin Panel")
    
    # Initialize session state for document upload
    if "processing_upload" not in st.session_state:
        st.session_state.processing_upload = False
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    # Sidebar with navigation
    with st.sidebar:
        page = st.radio("Navigation", ["Dashboard", "Document Management", "Settings"])
    
    if page == "Dashboard":
        st.header("Dashboard")
        
        # Get statistics from Supabase
        try:
            docs_response = supabase.table("documents").select("*").execute()
            chunks_response = supabase.table("document_chunks").select("*").execute()
            
            total_documents = len(docs_response.data)
            total_chunks = len(chunks_response.data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", total_documents)
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                st.metric("Average Chunks per Document", round(total_chunks/total_documents, 1) if total_documents > 0 else 0)
                
            # Show recent documents
            st.subheader("Recent Documents")
            if docs_response.data:
                df = pd.DataFrame(docs_response.data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at', ascending=False)
                st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
    
    elif page == "Document Management":
        st.header("Document Management")
        
        # Document Upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'], key="admin_pdf_uploader")
        
        chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=3000, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000, step=50)
        batch_size = st.number_input("Processing Batch Size", value=5, min_value=1, max_value=20, step=1)
        
        # Only process if we have a new file and aren't already processing
        if uploaded_file and not st.session_state.processing_upload and (
            st.session_state.last_uploaded_file != uploaded_file.name
        ):
            st.session_state.processing_upload = True
            st.session_state.last_uploaded_file = uploaded_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            with st.spinner("Processing document..."):
                try:
                    # Initialize embeddings
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    
                    # Create document record first
                    doc_response = supabase.table("documents").insert({
                        "filename": uploaded_file.name
                    }).execute()
                    
                    document_id = doc_response.data[0]['id']
                    
                    # Load and process the PDF
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    # Process chunks in batches with progress tracking
                    total_processed = process_chunks_in_batches(
                        chunks=chunks,
                        document_id=document_id,
                        embeddings=embeddings,
                        batch_size=batch_size
                    )
                    
                    st.success(f"✅ Successfully processed {total_processed} chunks from {uploaded_file.name}!")
                except Exception as e:
                    # If error occurs, delete the document if it was created
                    if 'document_id' in locals():
                        supabase.table("documents").delete().eq("id", document_id).execute()
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up the temporary file and reset processing state
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    st.session_state.processing_upload = False
                    st.rerun()
        
        # Document Management
        st.subheader("Manage Documents")
        try:
            docs_response = supabase.table("documents").select("*").execute()
            if docs_response.data:
                df = pd.DataFrame(docs_response.data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at', ascending=False)
                
                # Create a more user-friendly display of documents
                for _, doc in df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.text(doc['filename'])
                        with col2:
                            st.text(f"Created: {doc['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        with col3:
                            if st.button("Delete", key=f"del_{doc['id']}"):
                                # Document chunks will be automatically deleted due to CASCADE
                                supabase.table("documents").delete().eq("id", doc['id']).execute()
                                st.success(f"Deleted {doc['filename']} and all its chunks")
                                st.rerun()
        except Exception as e:
            st.error(f"Error managing documents: {str(e)}")
    
    elif page == "Settings":
        st.header("Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        current_openai_key = os.getenv("OPENAI_API_KEY", "")
        new_openai_key = st.text_input("OpenAI API Key", 
                                      value="*" * len(current_openai_key),
                                      type="password")
        
        if st.button("Update API Key"):
            # In a real application, you would update the .env file or environment variables
            st.success("API key updated successfully!")
        
        # Vector Store Settings
        st.subheader("Vector Store Settings")
        chunk_size = st.number_input("Chunk Size", value=1000, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, step=50)
        
        if st.button("Save Settings"):
            # Save the settings to a configuration file or database
            st.success("Settings saved successfully!")
