# RAG Assistant with ChatGPT-like Interface

A Retrieval Augmented Generation (RAG) application built with Streamlit, featuring a ChatGPT-like interface and an admin panel. The application uses Langchain for PDF processing and OpenAI embeddings stored in Supabase.

## Features

- ChatGPT-like user interface
- PDF document upload and processing
- Document retrieval using similarity search
- Admin panel for document management and system monitoring
- Secure authentication for admin access

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Supabase:
   - Create a new project at [Supabase](https://supabase.com)
   - Create a new table named 'documents' with the following SQL:
     ```sql
     create table documents (
       id bigint generated by default as identity primary key,
       content text,
       metadata jsonb,
       embedding vector(1536)
     );

     create index on documents using ivfflat (embedding vector_cosine_ops)
     with (lists = 100);
     ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration values

5. Run the main application:
   ```bash
   streamlit run app.py
   ```

6. Run the admin panel:
   ```bash
   streamlit run admin.py
   ```

## Usage

### Main Application
- Upload PDF documents using the sidebar
- Chat with the assistant about the uploaded documents
- The assistant will retrieve relevant information from the documents

### Admin Panel
- Access the admin panel using your configured credentials
- Monitor document statistics
- Manage uploaded documents
- Configure system settings

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service role API key
- `ADMIN_USERNAME`: Username for admin panel access
- `ADMIN_PASSWORD`: Password for admin panel access
