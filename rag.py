import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILE = os.path.join(
    BASE_DIR,
    "vector_db",
    "data",
    "nutrition.pdf"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)


# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------------------------------
# CREATE RAG DATABASE
# -------------------------------------------------

def create_rag():

    # Check PDF
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(
            f"Nutrition PDF not found at: {PDF_FILE}"
        )

    # Load PDF
    documents = PyPDFLoader(PDF_FILE).load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Create embeddings
    embedding = get_embeddings()

    # Create FAISS database
    db = FAISS.from_documents(
        chunks,
        embedding
    )

    # Save database
    db.save_local(VECTOR_DB_DIR)

    return len(chunks)


# -------------------------------------------------
# LOAD EXISTING RAG DATABASE
# -------------------------------------------------

def load_rag():

    # Check whether FAISS database exists
    index_file = os.path.join(
        VECTOR_DB_DIR,
        "index.faiss"
    )

    pickle_file = os.path.join(
        VECTOR_DB_DIR,
        "index.pkl"
    )

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at: {index_file}"
        )

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"FAISS pickle file not found at: {pickle_file}"
        )

    # Load embeddings
    embedding = get_embeddings()

    # Load FAISS database
    db = FAISS.load_local(
        VECTOR_DB_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )

    return db


# -------------------------------------------------
# SEARCH RAG DATABASE
# -------------------------------------------------

def search_rag(query, k=3):

    db = load_rag()

    results = db.similarity_search(
        query,
        k=k
    )

    return results
