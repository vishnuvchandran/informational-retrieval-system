import os
from typing import Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader

def load_document(file_path: Union[str, os.PathLike]) -> list[Document]:
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def load_document_from_uploadedfile(uploaded_file) -> list[Document]:
    
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load the document using the temporary file path
        documents = load_document(tmp_file_path)
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

    return documents