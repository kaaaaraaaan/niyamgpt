-- First, drop the existing document_chunks table
DROP TABLE IF EXISTS document_chunks;

-- Recreate the document_chunks table with 3072 dimensions
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(3072)
);
