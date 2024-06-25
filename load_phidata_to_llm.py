"""
This script processes phidata documentation and creates a question-answering system using language models and vector embeddings. Here's a detailed description of its functionality:

1. Import Libraries:
   - Imports necessary libraries for file handling, HTML parsing, text processing, and interactions with language models.

2. HTML Text Extraction:
   - Defines a function `extract_text_from_html` to remove HTML tags and extract plain text from HTML content.

3. Load Documentation Index:
   - Attempts to load a JSON file (`phidata_docs_index.json`) containing metadata about phidata documentation pages.

4. Process HTML Files:
   - Iterates through the index, reading HTML files from the 'phidata_docs' directory.
   - Extracts text content from each HTML file and stores it along with the source URL.
   - Keeps track of any missing files and reports them.

5. Text Splitting:
   - Uses RecursiveCharacterTextSplitter to break down the extracted text into smaller, manageable chunks.

6. Vector Embeddings:
   - Initializes OpenAIEmbeddings to convert text chunks into vector representations.

7. Vector Store Creation:
   - Creates a Chroma vector store from the text chunks, allowing for efficient similarity search.

8. Question-Answering System Setup:
   - Sets up a retrieval-based QA system using the Chroma vector store and ChatOpenAI language model.

9. Query Processing:
   - Defines a set of example queries about phidata.
   - Processes each query through the QA system and prints the results.

10. Error Handling:
    - Implements error handling throughout the script to catch and report issues during execution.

This script essentially creates a searchable knowledge base from phidata documentation, allowing users to ask questions and receive relevant answers based on the content of the documentation.
"""

# Import necessary libraries for file handling, HTML parsing, text processing, and language model interactions
import os
import json
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Function to extract plain text from HTML content, removing all HTML tags and formatting
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

# Load the index file containing information about phidata documentation pages
# This file should contain metadata about each documentation page, including its local path and URL
try:
    with open('phidata_docs_index.json', 'r') as f:
        index = json.load(f)
except FileNotFoundError:
    print("Error: phidata_docs_index.json not found. Make sure you're in the correct directory.")
    exit(1)

# Extract text from all HTML files listed in the index
# This process converts HTML documentation into plain text for further processing
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

# Print warnings for missing files to alert the user of any documentation pages that couldn't be processed
if missing_files:
    print(f"Warning: {len(missing_files)} files were not found:")
    for file in missing_files[:10]:  # Print first 10 missing files
        print(f"  - {file}")
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more.")

# Exit if no documents were loaded to prevent processing empty data
if not documents:
    print("Error: No documents were successfully loaded. Please check your phidata_docs folder.")
    exit(1)

print(f"Successfully loaded {len(documents)} documents.")

try:
    # Split text into smaller chunks for processing
    # This improves the efficiency of embedding and retrieval operations
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([doc['content'] for doc in documents], metadatas=[doc['metadata'] for doc in documents])

    # Initialize OpenAI embeddings for converting text into vector representations
    embeddings = OpenAIEmbeddings()

    # Create a Chroma vector store from the document chunks
    # This allows for efficient similarity search and retrieval of relevant text passages
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Set up the retrieval-based question-answering system
    # This combines the vector store with a language model to answer queries
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Define example queries to test the QA system
    # These queries cover common questions about phidata
    queries = [
        "How do I install phidata?",
        "What is phidata?",
        "How do I create a new project with phidata?",
        "What are the main features of phidata?"
    ]

    # Process each query and print the results
    # This demonstrates the capability of the QA system to answer various questions about phidata
    for query in queries:
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\nQuery: {query}")
            print("Answer:", result['result'])
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")

except Exception as e:
    print(f"An error occurred while setting up the QA system: {str(e)}")