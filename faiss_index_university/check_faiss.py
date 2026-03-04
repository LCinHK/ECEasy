from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your index
db = FAISS.load_local(
    "faiss_index_university",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("Number of vectors in index:", db.index.ntotal)