from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors
from utils.database import fetch_chunks, connect_db
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.llm_selection import get_llm, get_embedding_model
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_community.agent_toolkits import create_sql_agent
import streamlit as st


### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def process_query_new(query: str, model_choice):
    query_vector = get_query_embedding(query, model_choice)
    similar_chunk_ids = search_vectors(query_vector, n_results=10)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    embedding = get_embedding_model(model_choice)
    vectorstore = Chroma.from_documents(documents=relevant_chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()

    # Prepare the context by joining the relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    llm = get_llm(model_choice)
    
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    # qa_system_prompt = """You are an assistant for question-answering tasks. \
    # Use the following pieces of retrieved context to answer the question. \
    # If you don't know the answer, just say that you don't know. \
    # Use three sentences maximum and keep the answer concise.\

    # {context}"""


    qa_system_prompt = """You are an advanced assistant for question-answering tasks. Your goal is to provide accurate, comprehensive, and helpful responses based on the given context.

    Instructions:
    1. Carefully analyze all pieces of retrieved context provided below.
    2. Pay special attention to company names, abbreviations, and their full forms.
    3. Extract and synthesize relevant information to form a coherent and relevant answer to the question.
    4. If you find any information related to the question, even if it's not a complete answer, include it in your response.
    5. If you're unsure about any part of your answer, express your level of confidence.
    6. If you don't find any relevant information, clearly state that you don't have enough information to answer accurately.
    7. Provide a thorough answer without unnecessary length. Adjust the response length based on the complexity of the question and the available information.
    8. If appropriate, suggest follow-up questions or additional information that might be helpful.


    {context}"""


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "ragconv"}},
    )["answer"]

    return response


def process_db_query(query: str, model_choice):
    llm = get_llm(model_choice)
    db = connect_db()
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)

    # answer_prompt = PromptTemplate.from_template(
    #     """Given the following user question, corresponding PostgreSQL query, and PostgreSQL query result, answer the user question. If the query has a syntax error, provide the corrected query as well.

    #     Question: {question}
    #     PostgreSQL Query: {query}
    #     PostgreSQL Query Result: {result}
    #     Answer: """
    # )

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding PostgreSQL query, and PostgreSQL query result, answer the user question. If the query has a syntax error correct it. Do not include the syntax error and the query in the response.

        Question: {question}
        PostgreSQL Query: {query}
        PostgreSQL Query Result: {result}
        Answer: """
    )


    answer = answer_prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    response = chain.invoke({"question": query})
    return response


def process_query(query: str, model_choice):
    
    query_vector = get_query_embedding(query, model_choice)
    similar_chunk_ids = search_vectors(query_vector, n_results=5)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    llm = get_llm(model_choice)
    
    # Prepare the context by joining the relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    st.write(context)
    template = """You are an advanced assistant for question-answering tasks. Your goal is to provide accurate, comprehensive, and helpful responses based on the given context.

    Instructions:
    1. Carefully analyze all pieces of retrieved context provided below.
    2. Pay special attention to company names, abbreviations, and their full forms.
    3. Extract and synthesize relevant information to form a coherent and relevant answer to the question.
    4. If you find any information related to the question, even if it's not a complete answer, include it in your response.
    5. If you're unsure about any part of your answer, express your level of confidence.
    6. If you don't find any relevant information, clearly state that you don't have enough information to answer accurately.
    7. Provide a thorough answer without unnecessary length. Adjust the response length based on the complexity of the question and the available information.
    8. If appropriate, suggest follow-up questions or additional information that might be helpful.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": lambda q: context, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)
    return response


def process_db_agent(query: str, model_choice):
    llm = get_llm(model_choice)
    db = connect_db()
    
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    response = agent_executor.invoke(query)
    return response