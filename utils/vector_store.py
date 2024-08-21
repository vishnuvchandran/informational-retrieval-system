import chromadb
import uuid

# chroma_client = Client(Settings(persist_directory="./chroma_db"))
# collection = chroma_client.get_or_create_collection(name=collection_name)

client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "document_chunks"
collection = client.get_or_create_collection(name=collection_name)


def store_vectors(chunks, vectors):
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": chunk.metadata.get("source", ""), "page": chunk.metadata.get("page", 0)} for chunk in chunks]
    
    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas
    )
    return ids
 

def search_vectors(query_vector, n_results=10):
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results['ids'][0]