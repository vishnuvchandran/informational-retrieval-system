import streamlit as st
from utils.document_loader import load_document_from_uploadedfile
from utils.text_splitter import split_text
from utils.embeddings import get_embeddings
from utils.vector_store import store_vectors
from utils.query_processing import process_query, process_db_query, process_db_agent, process_db_vector, process_query_history
from utils.database import store_chunks, fetch_chunks
from langchain.chains import RetrievalQA



def main():
    st.set_page_config(layout="wide")

    # Application type selection dropdown
    app_type = st.sidebar.selectbox(
        "Select the Application Behaviour:",
        ("RAG", "RAG-conv-histry", "DB-QnA", "DB-QnA-agent", "DB-QnA-vector")
    )

    # Model selection dropdown
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("google", "openai")
    )
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Upload")
        uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Document processing pipeline
                    doc = load_document_from_uploadedfile(uploaded_file)
                    chunks = split_text(doc)
                    vectors = get_embeddings(chunks, model_choice)
                    chunk_ids = store_vectors(chunks, vectors)
                    store_chunks(chunks, chunk_ids)
                st.success("Document processed successfully!")

    # Main chat interface
    st.title("RAG Document Q&A System")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the query and generate response
        with st.spinner("Generating response..."):
            if app_type == 'RAG':
                response = process_query(prompt, model_choice)
            elif app_type == 'RAG-conv-histry':
                response = process_query_history(prompt, model_choice)
            elif app_type == 'DB-QnA':    
                response = process_db_query(prompt, model_choice)
            elif app_type == 'DB-QnA-agent':
                response = process_db_agent(prompt, model_choice)
            elif app_type == 'DB-QnA-vector':
                response =  process_db_vector(prompt, model_choice)
            else:
                response = process_query(prompt, model_choice)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()