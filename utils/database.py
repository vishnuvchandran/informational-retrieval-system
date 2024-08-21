import psycopg2
from psycopg2.extras import execute_batch
import json
from dotenv import load_dotenv
import os
from langchain_core.documents import Document


load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

def store_chunks(chunks, chunk_ids):
    conn = get_db_connection()

    with conn.cursor() as cur:
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB
            )
        """)
        
        # Prepare data for insertion
        data = [(id, chunk.page_content, json.dumps(chunk.metadata)) for id, chunk in zip(chunk_ids, chunks)]
        
        # Insert data
        execute_batch(cur, """
            INSERT INTO document_chunks (id, content, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
        """, data)
    
    conn.commit()
    conn.close()

def fetch_chunks(chunk_ids):
    conn = get_db_connection()
    with conn.cursor() as cur:
        placeholders = ','.join(['%s'] * len(chunk_ids))
        cur.execute(f"SELECT id, content, metadata FROM document_chunks WHERE id IN ({placeholders})", chunk_ids)
        results = cur.fetchall()
    
    chunks = [Document(page_content=content, metadata=metadata) for _, content, metadata in results]
    conn.close()
    return chunks