from langchain_google_genai import GoogleGenerativeAI
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List

# Dictionary to map model choices to their corresponding API keys
API_KEYS = {}

# Function to load environment variables and set API key
def set_api_key(model_choice):
    load_dotenv()
    global API_KEYS

    if not API_KEYS:  # Load the keys only once
        API_KEYS = {
            "google": os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }

    if model_choice in API_KEYS:
        os.environ[f"{model_choice.upper()}_API_KEY"] = API_KEYS[model_choice]
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

# Function to get LLM based on model choice
def get_llm(model_choice):
    set_api_key(model_choice)
    if model_choice == "google":
        return GoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    elif model_choice == "openai":
        return ChatOpenAI(model="gpt-3.5-turbo-0125")
    
# Function to get embedding model based on model choice
def get_embedding_model(model_choice):
    set_api_key(model_choice)
    if model_choice == "google":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif model_choice == "openai":
        return OpenAIEmbeddings()
    

class MyVertexAIEmbeddings(VertexAIEmbeddings, Embeddings):
    model_name = 'textembedding-gecko'
    max_batch_size = 5
    
    def embed_segments(self, segments: List) -> List:
        embeddings = []
        for i in tqdm(range(0, len(segments), self.max_batch_size)):
            batch = segments[i: i+self.max_batch_size]
            embeddings.extend(self.client.get_embeddings(batch))
        return [embedding.values for embedding in embeddings]
    
    def embed_query(self, query: str) -> List:
        embeddings = self.client.get_embeddings([query])
        return embeddings[0].values
    

def get_vertex_embedding():
    return  MyVertexAIEmbeddings()
