import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama 
import faiss


def get_pdf_files_from_data_folder(folder_path):
    """
    This function takes a path to a folder (folder_path) and returns
    a list of all the PDF files in that folder. The path to each PDF
    file is an absolute path.

    :param folder_path: The path to the folder that contains the
                        PDF files.
    :return: A list of paths to all the PDF files in the folder.
    """
    # Loop over all the files in the folder
    pdf_files = []
    for file in os.listdir(folder_path):
        # Check whether the current file is a PDF
        if file.endswith('.pdf'):
            # If it is, add the absolute path to the list of PDF files
            pdf_files.append(os.path.join(folder_path, file))
    # Return the list of PDF files
    return pdf_files
def load_pdf_from_folder(folder_path):
    """
    This function takes a path to a folder as input (folder_path) and
    returns a list of documents, where each document is a loaded PDF
    file from the folder.

    The function works by first getting the list of all PDF files in
    the folder, and then looping over them and loading each one into
    a document. The list of documents is then returned.

    :param folder_path: The path to the folder that contains the PDF
                        files.
    :return: A list of documents, where each document is a loaded PDF
             file from the folder.
    """
    # Get the list of all PDF files in the folder
    pdf_files = get_pdf_files_from_data_folder(folder_path)
    # Initialize an empty list to contain all the documents
    documents = []
    # Loop over all the PDF files
    for pdf_file in pdf_files:
        # Create a PyPDFLoader object from the PDF file
        loader = PyPDFLoader(pdf_file)
        # Use the loader to load the PDF file into a document
        document = loader.load()
        # Add the document to the list of documents
        documents.append(document)
    # Return the list of documents
    return documents
def split_documents(documents):
    """
    This function takes a list of documents as input and splits
    each document into chunks of a fixed size. The chunks are
    then returned as a list.

    The splitting is done using the RecursiveCharacterTextSplitter
    class from the langchain_text_splitters module. This class
    takes two parameters: chunk_size and chunk_overlap.

    The chunk_size parameter determines the maximum size of each
    chunk. For example, if chunk_size is set to 1000, then each
    chunk will be at most 1000 characters long.

    The chunk_overlap parameter determines how much the chunks
    should overlap with each other. For example, if chunk_overlap
    is set to 100, then the last 100 characters of each chunk
    will be repeated as the first 100 characters of the next
    chunk.

    The function works by first creating a RecursiveCharacterTextSplitter
    object with the specified chunk_size and chunk_overlap parameters.
    It then loops over all the documents, and for each document, it
    uses the text_splitter to split the document into chunks. The
    chunks are then added to a list, which is returned at the end.

    :param documents: A list of documents to be split.
    :return: A list of chunks, where each chunk is a subset of
             one of the documents.
    """
    # Create a RecursiveCharacterTextSplitter object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    # Initialize an empty list to contain all the chunks
    chunk_documents = []
    # Loop over all the documents
    for document in documents:
        # Split the document into chunks
        chunks = text_splitter.split_documents(document)
        # Add the chunks to the list of chunks
        chunk_documents.extend(chunks)
    # Return the list of chunks
    return chunk_documents
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
def create_embeddings(chunk_documents):
    """
    This function takes a list of chunks as input and returns
    a numpy array of embeddings, where each embedding corresponds
    to one of the chunks.

    The function works by looping over all the chunks, and for
    each chunk, it uses the embedding_model to create an embedding
    for the chunk. The embedding is then added to the list of
    embeddings, which is returned at the end.

    :param chunk_documents: A list of chunks to be embedded.
    :return: A numpy array of embeddings, where each embedding
             corresponds to one of the chunks.
    """
    # Initialize an empty list to contain all the embeddings
    embeddings = []
    # Loop over all the chunks
    for chunk in chunk_documents:
        # Use the embedding_model to create an embedding for the chunk
        chunk_embedding = embedding_model.encode(chunk.page_content)
        # Add the embedding to the list of embeddings
        embeddings.append(chunk_embedding)
    # Convert the list of embeddings to a numpy array
    embeddings = np.array(embeddings)
    # Return the numpy array of embeddings
    return embeddings

def create_faiss_index(embeddings):
    """
    This function takes a numpy array of embeddings as input and
    returns a FAISS index that can be used to search for the nearest
    neighbors of a given query.

    The function works by creating an IndexFlatL2 object, which is a
    type of FAISS index that supports efficient nearest neighbor
    search. The index is then populated with the given embeddings.

    :param embeddings: A numpy array of embeddings to be indexed.
    :return: A FAISS index that can be used to search for the nearest
             neighbors of a given query.
    """
    dimension = embeddings.shape[1]
    # Create an IndexFlatL2 object, which is a type of FAISS index that
    # supports efficient nearest neighbor search.
    faiss_index = faiss.IndexFlatL2(dimension)
    # Populate the index with the given embeddings.
    faiss_index.add(embeddings)
    # Return the FAISS index.
    return faiss_index
def retrieve_context(query, faiss_index, chunk_documents):
    """
    This function takes a query and a FAISS index as input, and returns
    a string that contains the context relevant to the query.

    The function works by first encoding the query into an embedding
    using the embedding_model. It then uses the FAISS index to search
    for the k=3 nearest neighbors to the query. The text of the
    relevant documents is then extracted from the chunk_documents list
    and joined together into a single string, which is returned as the
    context.

    :param query: The query to retrieve context for.
    :param faiss_index: The FAISS index to use for searching.
    :param chunk_documents: The list of documents to retrieve context from.
    :return: A string that contains the context relevant to the query.
    """
    # Encode the query into an embedding using the embedding_model
    query_embedding = embedding_model.encode([query])
    # Search for the k=3 nearest neighbors to the query using the FAISS index
    D, I = faiss_index.search(query_embedding, k=3)
    # Extract the text of the relevant documents from the chunk_documents list
    relevant_docs = [chunk_documents[i].page_content for i in I[0]]
    # Join the relevant documents together into a single string
    context = "\n".join(relevant_docs)
    # Return the context
    return context

def create_augmented_prompt(query, context):
    """
    This function takes a query and a context string as input and
    returns an augmented prompt that can be used to generate a helpful
    response to the query.

    The function works by formatting the context and query into a
    string that can be used as a prompt to a language model.

    :param query: The query to generate a response for.
    :param context: A string containing the context relevant to the query.
    :return: An augmented prompt that can be used to generate a helpful
             response to the query.
    """
    prompt_template = """
    You are a helpful AI language model. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    Helpful Answer:"""

    return prompt_template.format(context=context,question=query)
    
def run_rag_agent(folder_path):
    """
    This function runs the RAG agent.

    The agent first loads all the PDF files from the specified folder
    into a list of documents. It then splits each document into chunks
    of a fixed size, and creates a FAISS index from the chunks.

    The agent then enters a user interaction loop. In this loop, the
    user is prompted to enter a question, and the agent responds with
    a helpful answer to the question. The loop continues until the user
    types 'exit' to quit.

    :param folder_path: The path to the folder that contains the PDF
                        files to be indexed.
    """
    print("Loading PDF files from", folder_path, "into a list of documents...")
    documents = load_pdf_from_folder(folder_path)
    print("Splitting each document into chunks of a fixed size...")
    chunk_documents = split_documents(documents)
    print("Creating a numpy array of embeddings from the chunks...")
    embeddings = create_embeddings(chunk_documents)
    print("Creating a FAISS index from the embeddings...")
    faiss_index = create_faiss_index(embeddings)

    # User interaction loop
    while True:
        print("Prompting the user to enter a question...")
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting the RAG agent. Goodbye!")
            break
        print("Retrieving context relevant to the question...")
        context = retrieve_context(question, faiss_index, chunk_documents)
        print("Creating an augmented prompt for the language model...")
        # This line creates an augmented prompt for the language model.
        # The augmented prompt is then passed to the language model, which
        # generates a helpful response to the question.
        prompt = create_augmented_prompt(question, context)
        print("Generating a helpful response to the question...")
        # This line generates a helpful response to the question. The
        # response is then printed to the console.
        response = ollama.generate(model='mistral', prompt=prompt)
        
        print("\nAnswer:\n", response['response'])
        print("\n-----------------------------\n")

    
    
        
run_rag_agent('data')