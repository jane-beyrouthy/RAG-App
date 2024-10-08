# Required Imports
import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.ollama import Ollama
from langchain.chains import RetrievalQA

# Step 1: Extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Step 2: Extract text from multiple PDFs
def extract_text_from_multiple_pdfs(pdf_paths):
    combined_text = ""
    for pdf_path in pdf_paths:
        combined_text += extract_text_from_pdf(pdf_path) + "\n"
    return combined_text

# Step 3: Split text into chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# Step 4: Generate embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Step 5: Create FAISS vector store
def create_vector_store(docs, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Step 6: Load MistralAI using Ollama
def load_mistral_llm():
    mistral_llm = Ollama(model="mistral")
    return mistral_llm

# Step 7: Create RetrievalQA Chain
def create_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

# Step 8: Query the system
def query_rag_agent(qa_chain, query):
    response = qa_chain.run(query)
    return response

# Step 9: Get all PDF files from the 'data' folder
def get_pdf_files_from_data_folder(folder_path):
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    return pdf_files

# Main function to run the RAG agent
def run_rag_agent(data_folder):
    # Step 9: Get all PDF files from the 'data' folder
    pdf_paths = get_pdf_files_from_data_folder(data_folder)
    
    # Step 2: Extract text from multiple PDFs
    combined_text = extract_text_from_multiple_pdfs(pdf_paths)
    
    # Step 3: Split text into chunks
    docs = split_text_into_chunks(combined_text)
    
    # Step 4: Generate embeddings
    embeddings = create_embeddings()
    
    # Step 5: Create FAISS vector store
    vector_store = create_vector_store(docs, embeddings)
    
    # Step 6: Load MistralAI via Ollama
    mistral_llm = load_mistral_llm()
    
    # Step 7: Create RetrievalQA chain
    qa_chain = create_qa_chain(vector_store, mistral_llm)
    
    # Step 8: Loop to allow multiple queries
    while True:
        query = input("Please enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the RAG agent.")
            break
        response = query_rag_agent(qa_chain, query)
        print("Response:", response)

# Example usage
if __name__ == "__main__":
    # Path to the 'data' folder containing multiple PDF files
    data_folder = "../data"  # Relative path to 'data' folder from the 'src/' folder
    
    # Run the RAG agent with PDFs from the 'data' folder
    run_rag_agent(data_folder)
