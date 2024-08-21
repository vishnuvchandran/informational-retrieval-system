from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors
from utils.database import fetch_chunks
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.llm_selection import get_llm, get_embedding_model
from langchain_chroma import Chroma
import streamlit as st


def process_query(query: str, model_choice):
    query_vector = get_query_embedding(query, model_choice)
    similar_chunk_ids = search_vectors(query_vector, n_results=10)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    embedding = get_embedding_model(model_choice)
    vectorstore = Chroma.from_documents(documents=relevant_chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()
    # Prepare the context by joining the relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    st.write(context)
    def format_docs(relevant_chunks):
        return "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    llm = get_llm(model_choice)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)
    return response