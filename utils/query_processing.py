from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors, search_documents
from utils.database import fetch_chunks, connect_db
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.llm_selection import get_llm, get_embedding_model, get_vertex_embedding
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
import ast
import re
from utils.external_db import EnhancedRAGSystem
from langchain.document_loaders import JSONLoader
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
import json
import streamlit as st
import os



### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def process_query_history(query: str, model_choice):
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
    # relevant_chunks = search_documents(query_vector, query, n_results=5)
    
    llm = get_llm(model_choice)
    
    # Prepare the context by joining the relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
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
    embedding = get_embedding_model(model_choice)

    def query_as_list(db, query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return res


    proper_nouns = query_as_list(db, "SELECT Name FROM Artist")
    proper_nouns += query_as_list(db, "SELECT Title FROM Album")
    proper_nouns += query_as_list(db, "SELECT Name FROM Genre")
    len(proper_nouns)
    proper_nouns[:5]

    vector_db = FAISS.from_texts(proper_nouns, embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})

    system = """You are a SQLite expert. Given an input question, create a syntactically \
    correct SQLite query to run. Unless otherwise specificed, do not return more than \
    {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nHere is a non-exhaustive \
    list of possible feature values. If filtering on a feature value make sure to check its spelling \
    against this list first:\n\n{proper_nouns}"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    retriever_chain = (
        itemgetter("question")
        | retriever
        | (lambda docs: "\n".join(doc.page_content for doc in docs))
    )
    chain = RunnablePassthrough.assign(proper_nouns=retriever_chain) | query_chain

    response = chain.invoke({"question": query})
    return response


def process_db_vector(query: str, model_choice):
    db = connect_db()
    llm = get_llm(model_choice)
    rag_system = EnhancedRAGSystem(db, llm)
    rag_system.initialize()
    response = rag_system.process_query(query)
    return response


def process_text_to_sql(query: str):
    # embedding = get_vertex_embedding()
    embedding = get_embedding_model('google')
    llm = get_llm('google')
    pgdb = connect_db()
    documents = JSONLoader(file_path='./schemanew.jsonl', jq_schema='.', text_content=False, json_lines=True).load()
    db = FAISS.from_documents(documents=documents, embedding=embedding)
    
    retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 1})
    matched_documents = retriever.get_relevant_documents(query=query)

    matched_tables = []

    for document in matched_documents:
        page_content = document.page_content
        page_content = json.loads(page_content)
        table_name = page_content['table_name']
        matched_tables.append(f'{table_name}')

    search_kwargs = {
        'k': 20
    }
    
    retriever = db.as_retriever(search_type='similarity', search_kwargs=search_kwargs)
    matched_columns = retriever.get_relevant_documents(query=query)

    matched_columns_filtered = []

    for i, column in enumerate(matched_columns):
        page_content = json.loads(column.page_content)
        matched_columns_filtered.append(page_content)
    
    matched_columns_cleaned = []
    
    for table in matched_columns_filtered:
        table_name = table['table_name']
        for column in table['columns']:
            column_name = column['name']
            data_type = column['type']
            matched_columns_cleaned.append(f'table_name={table_name}|column_name={column_name}|data_type={data_type}')
    
    matched_columns_cleaned = '\n'.join(matched_columns_cleaned)

    messages = []

    template = "You are a SQL master expert capable of writing complex SQL queries in PostgreSQL."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    messages.append(system_message_prompt)

    human_template = """Given the following inputs:
    USER_QUERY:
    --
    {query}
    --
    MATCHED_SCHEMA: 
    --
    {matched_schema}
    --
    Please construct a SQL query using the MATCHED_SCHEMA and the USER_QUERY provided above.

    IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. DO NOT USE any other column names outside of this. 
    IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA.
    NOTE: Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed. 
    """

    human_message = HumanMessagePromptTemplate.from_template(human_template)
    messages.append(human_message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    request = chat_prompt.format_prompt(query=query, matched_schema=matched_columns_cleaned).to_messages()
    
    response = llm.invoke(request)
    sql_query = '\n'.join(response.strip().split('\n')[1:-1])

    result = pgdb.run(sql_query)

    final_template = """
    Here is the result of your query:

    User Query:
    {user_query}

    Generated SQL Query:
    {sql_query}

    Query Result:
    {result}
    """

    final_response = llm.invoke(
        final_template.format(user_query=query, sql_query=sql_query, result=result)
    )
    
    st.write(response)
    return final_response
