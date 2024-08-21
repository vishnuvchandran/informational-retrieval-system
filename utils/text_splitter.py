from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_text(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split the input documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): A list of Document objects to be split.
        chunk_size (int): The size of each chunk in characters. Default is 1000.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.
    
    Returns:
        List[Document]: A list of Document objects representing the split chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_documents([doc]))
    
    return chunks