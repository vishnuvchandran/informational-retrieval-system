from langchain.docstore.document import Document
from typing import List, Union
from utils.llm_selection import get_embedding_model


def get_embeddings(texts: Union[List[str], List[Document]], model_choice) -> List[List[float]]:
    if isinstance(texts[0], Document):
        # If input is a list of Documents, extract the page_content
        texts = [doc.page_content for doc in texts]
    
    # Generate embeddings
    embeddings_model = get_embedding_model(model_choice)
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def get_query_embedding(query: str, model_choice) -> List[float]:
    embeddings_model = get_embedding_model(model_choice)
    return embeddings_model.embed_query(query)