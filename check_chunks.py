from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(
        "faiss_index_university",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Total chunks in index: {db.index.ntotal}")
except Exception as e:
    print(f"Error loading index: {e}")