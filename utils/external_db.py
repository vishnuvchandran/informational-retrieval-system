import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.tools import QuerySQLDataBaseTool
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from utils.database import get_db_connection
import streamlit as st

class EnhancedRAGSystem:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.execute_query = QuerySQLDataBaseTool(db=self.db)
        self.write_query = create_sql_query_chain(self.llm, self.db)
        self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.id_to_data = {}


    def get_table_info(self):
        table_info = {}
        query = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
                
        for table_name, column_name in results:
            if table_name not in table_info:
                table_info[table_name] = []
            table_info[table_name].append(column_name)
        
        return table_info


    def vectorize_schema(self):
        table_info = self.get_table_info()
        vectors = []
        for table, columns in table_info.items():
            schema_text = f"Table: {table}\nColumns: {', '.join(columns)}"
            vector = self.vectorizer.encode([schema_text])[0]
            vectors.append(vector)
            self.id_to_data[len(vectors) - 1] = {"table": table, "columns": columns}

        self.index = faiss.IndexFlatL2(vectors[0].shape[0])
        self.index.add(np.array(vectors))


    def get_relevant_tables(self, query, k=3):
        query_vector = self.vectorizer.encode([query])
        _, indices = self.index.search(query_vector, k)
        return [self.id_to_data[i] for i in indices[0]]
    

    def process_query(self, user_query):
        # Get relevant tables based on the query
        relevant_tables = self.get_relevant_tables(user_query)
        
        # Create a context string from relevant tables
        context = "Relevant tables and columns:\n"
        for table_info in relevant_tables:
            context += f"Table: {table_info['table']}\n"
            context += f"Columns: {', '.join([col for col in table_info['columns']])}\n\n"


        # Enhanced prompt with context
        enhanced_prompt = PromptTemplate.from_template(
            """Given the following context, user question, corresponding PostgreSQL query, and PostgreSQL query result, answer the user question. If the query has a syntax error, correct it. Do not include the syntax error and the query in the response.

            Context:
            {context}

            Question: {question}
            PostgreSQL Query: {query}
            PostgreSQL Query Result: {result}
            Answer: """
        )

        answer = enhanced_prompt | self.llm | StrOutputParser()

        chain = (
            RunnablePassthrough.assign(query=self.write_query).assign(
                result=itemgetter("query") | self.execute_query
            )
            | answer
        )

        response = chain.invoke({"question": user_query, "context": context})
        return response


    def initialize(self):
        self.vectorize_schema()
