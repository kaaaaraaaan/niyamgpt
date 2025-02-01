-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Drop existing function and table
drop function if exists match_documents(vector(1536), int, float);
drop table if exists documents;
drop table if exists document_chunks;

-- Create a table for documents
create table if not exists documents (
    id uuid default gen_random_uuid() primary key,
    filename text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create a table for document chunks
create table if not exists document_chunks (
    id uuid default gen_random_uuid() primary key,
    document_id uuid references documents(id) on delete cascade,
    content text not null,
    metadata jsonb,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create a function to match document chunks based on embedding similarity
create or replace function match_documents(
    query_embedding vector(1536),
    match_count int default 4,
    match_threshold float default 0.8
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        document_chunks.id,
        document_chunks.content,
        document_chunks.metadata,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from document_chunks
    where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
end;
$$;
