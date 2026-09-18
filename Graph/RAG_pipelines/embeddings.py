from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

EMBEDDING_MODEL=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

@traceable(name="creating vector embeddings")
def create_embeddings(chunks:list):
        database=Chroma.from_documents(
                documents=chunks,
                embedding=EMBEDDING_MODEL,
                persist_directory="chromadb"
        )
        return database

@traceable(name="loading vector embeddings")
def load_embeedings():
        pass