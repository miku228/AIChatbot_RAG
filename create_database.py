
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_chroma import Chroma
import os
import shutil
from dotenv import load_dotenv
import openai 


# Load environment variables from .env file that is in root directory of this project
load_dotenv()
# Set OpenAI API key 
# openai.api_key = os.esnviron['OPENAI_API_KEY']


CHROMA_PATH = "chroma"
# DATA_PATH = "data/books"
DATA_PATH = "data/resume"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chuncks = split_text(documents)
    save_to_chorma(chuncks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    # print(f"documents * {documents}")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        # chunk_size=1000,
        chunk_size=10,
        chunk_overlap=5,
        length_function=len,
        add_start_index = True,
    )
    chuncks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document info {len(chuncks)} chuncks")
    return chuncks


def save_to_chorma(chunks: list[Document]):
     # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        print("Deleting chunks directory...")
        shutil.rmtree(CHROMA_PATH)

    print(f"Saving {len(chunks)} chunks to Chroma database....")

    # Create a new DB from the documents.
    try:
        # Try using HuggingFaceEmbeddings with a simpler model
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        ## ******* ADD Fallback logic 

    # # Create a new DB from the documents.
    # db = Chroma.from_documents(
    #     chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    # )
    # db.persist()
    # print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


    print(f"Using embedding function: {embedding_function}")
    # Create a new DB from the documents - remove the persist() call
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    print(f"Saving {len(chunks)} chunks to Chroma database at {CHROMA_PATH}")

    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")





if __name__ == "__main__":
    main()