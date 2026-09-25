import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF location
PDF_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nutrition.pdf"
)

# FAISS database location
VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_rag():

    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(
            f"Nutrition PDF not found: {PDF_FILE}"
        )

    documents = PyPDFLoader(PDF_FILE).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(VECTOR_DB_DIR)

    return len(chunks)


def load_rag():

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
            f"index.faiss not found: {index_file}"
        )

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"index.pkl not found: {pickle_file}"
        )

    embeddings = get_embeddings()

    db = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db
