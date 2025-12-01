import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# FIXED: Using the new official library import
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_data(pdf_path):
    print(f"📖 Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"🧠 Creating memory database with {len(all_splits)} chunks...")
    
    # FIXED: Using the model from the new library
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save to a local folder named 'vector_db'
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_model,
        persist_directory="./vector_db"
    )
    print("✅ Ingestion complete! Memory saved to 'vector_db' folder.")

if __name__ == "__main__":
    ingest_data("homeopathy_book.pdf")