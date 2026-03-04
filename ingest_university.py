from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Folders
DATA_PATH = Path("knowledge/university_life")      # Put your .txt / .docx / .pdf files here
INDEX_PATH = "faiss_index_university"              # Where FAISS saves the index

def load_all_documents(data_path: Path):
    """Load .txt, .docx, and .pdf files using separate loaders and combine."""
    all_docs = []

    # Load .txt files
    txt_loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        recursive=True,
        silent_errors=True
    )
    all_docs.extend(txt_loader.load())

    # Load .docx files
    docx_loader = DirectoryLoader(
        data_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        recursive=True,
        silent_errors=True
    )
    all_docs.extend(docx_loader.load())

    # Load .pdf files
    pdf_loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        recursive=True,
        silent_errors=True
    )
    all_docs.extend(pdf_loader.load())

    return all_docs

def main():
    if not DATA_PATH.exists() or not any(DATA_PATH.iterdir()):
        print(f"Error: Folder '{DATA_PATH}' is empty or doesn't exist.")
        print("→ Create it and add at least one .txt / .docx / .pdf file")
        return

    print("Loading documents...")
    docs = load_all_documents(DATA_PATH)
    print(f"Loaded {len(docs)} documents in total")

    if len(docs) == 0:
        print("No documents loaded → nothing to index. Add files and retry.")
        return

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True                # Helpful for tracing back to source
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",      # Fast & decent quality
        model_kwargs={"device": "cpu"},     # Change to "mps" if you want Apple Silicon acceleration
    )

    print("Embedding and saving index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vectorstore.save_local(INDEX_PATH)
    print(f"Saved FAISS index to: {INDEX_PATH}")

if __name__ == "__main__":
    main()