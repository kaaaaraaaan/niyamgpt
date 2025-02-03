import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def process_chunks_in_batches(chunks: List[Document], document_id: str, embeddings: OpenAIEmbeddings = None, batch_size: int = 5):
    """Process document chunks in batches with progress tracking."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        status_text.text(f"Processing {min(i+batch_size, total_chunks)}/{total_chunks} chunks")
        
        # Process each chunk in the batch
        for chunk in batch:
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    chunk_embedding = embeddings.embed_query(chunk.page_content)
                    response = supabase.table("document_chunks").insert({
                        "document_id": document_id,
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "embedding": chunk_embedding
                    }).execute()
                    time.sleep(0.5)  # Add a small delay between chunks
                    break  # If successful, break the retry loop
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        st.error(f"Failed to process chunk: {str(e)}")
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

# Initialize session state for document upload
if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if check_password():
    st.title("RAG Admin Panel")
    
    # Sidebar with navigation
    with st.sidebar:
        page = st.radio("Navigation", ["Dashboard", "Upload Documents", "Manage Documents"])
    
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
    
    elif page == "Upload Documents":
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'], key="admin_pdf_uploader")
        # Set optimal defaults for legal document processing
        chunk_size = 640  # Optimal size between 512-768 tokens
        chunk_overlap = 128  # ~20% overlap
        batch_size = 5

        if uploaded_file:
            st.write(f"Selected file: {uploaded_file.name}")
            if st.button("Process Document"):
                st.session_state.processing_upload = True
                st.session_state.last_uploaded_file = uploaded_file.name
                
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                with st.spinner("Processing document..."):
                    try:
                        from langchain_community.document_loaders.pdf import PyPDFLoader
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        
                        # Load PDF
                        loader = PyPDFLoader(tmp_file_path)
                        documents = loader.load()
                        
                        # Create document record
                        doc_response = supabase.table("documents").insert({"filename": uploaded_file.name}).execute()
                        document_id = doc_response.data[0]['id']
                        
                        # Split and process chunks
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
                        chunks = text_splitter.split_documents(documents)
                        
                        total_processed = process_chunks_in_batches(chunks=chunks, document_id=document_id, batch_size=batch_size)
                        st.success(f"Successfully processed document: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        if 'document_id' in locals():
                            st.write("Cleaning up failed document record...")
                            supabase.table("documents").delete().eq("id", document_id).execute()
                    finally:
                        import os
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
                        st.session_state.processing_upload = False
                        st.rerun()
        
    elif page == "Manage Documents":
        st.header("Manage Documents")
        try:
            import pandas as pd
            docs_response = supabase.table("documents").select("*").execute()
            if docs_response.data:
                df = pd.DataFrame(docs_response.data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at', ascending=False)
                for _, doc in df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.text(doc['filename'])
                        with col2:
                            st.text(f"Created: {doc['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        with col3:
                            if st.button("Delete", key=f"del_{doc['id']}"):
                                supabase.table("documents").delete().eq("id", doc['id']).execute()
                                st.success(f"Deleted {doc['filename']} and all its chunks")
                                st.rerun()
        except Exception as e:
            st.error(f"Error managing documents: {str(e)}")
