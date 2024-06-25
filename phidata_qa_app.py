import streamlit as st
import os
import json
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Function to extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

# Function to load and process documents
@st.cache_resource
def load_qa_system():
    try:
        with open('phidata_docs_index.json', 'r') as f:
            index = json.load(f)
    except FileNotFoundError:
        st.error("Error: phidata_docs_index.json not found. Make sure you're in the correct directory.")
        return None, 0, 0

    documents = []
    missing_files = []
    for page in index:
        file_path = os.path.join('phidata_docs', page['local_path'])
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, 'index.html')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            text_content = extract_text_from_html(html_content)
            documents.append({"content": text_content, "metadata": {"source": page['url']}})
        else:
            missing_files.append(file_path)

    if not documents:
        st.error("Error: No documents were successfully loaded. Please check your phidata_docs folder.")
        return None, 0, len(missing_files)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([doc['content'] for doc in documents], metadatas=[doc['metadata'] for doc in documents])

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return qa_chain, len(documents), len(missing_files)

# Function to get answer
def get_answer(qa_chain, question):
    try:
        result = qa_chain.invoke({"query": question})
        return result['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
st.set_page_config(page_title="Phidata Documentation Q&A", layout="wide")

# Custom CSS for better styling and fixing the text input issue
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color: #31333F;  /* Ensuring text is visible */
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .message {
        width: 100%;
        padding: 0 1.5rem;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Phidata Documentation Q&A")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load QA system
qa_chain, doc_count, missing_count = load_qa_system()

if qa_chain:
    st.write(f"Successfully loaded {doc_count} documents. {missing_count} files were not found.")

    # Sidebar with example questions
    st.sidebar.header("Example Questions")
    example_questions = [
        "How do I install phidata?",
        "What is phidata?",
        "How do I create a new project with phidata?",
        "What are the main features of phidata?",
        "How can I use phidata for data processing?",
        "What are the best practices for using phidata?"
    ]
    for question in example_questions:
        if st.sidebar.button(question):
            answer = get_answer(qa_chain, question)
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("bot", answer))

    # Main chat interface
    st.header("Chat with Phidata Assistant")

    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="chat-message user"><div class="message">User: {message}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot"><div class="message">Assistant: {message}</div></div>', unsafe_allow_html=True)

    # User input
    user_question = st.text_input("Ask a question about phidata:", key="user_input")

    if user_question:
        if st.button("Send", key="send_button"):
            answer = get_answer(qa_chain, user_question)
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", answer))
            st.experimental_rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

else:
    st.error("Failed to initialize the QA system. Please check the console for more details.")

# Footer
st.markdown("---")
st.markdown("Powered by Phidata and Streamlit")