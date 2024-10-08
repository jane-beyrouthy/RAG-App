import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    
def main():
    folder_path = 'data'
    documents = load_pdf_from_folder(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for document in documents:
        chunk_documents = text_splitter.split_documents(document)
        chunks.extend(chunk_documents)
        
    print(chunks[0])
    
    
        
main()