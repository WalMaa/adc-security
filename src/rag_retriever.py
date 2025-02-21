from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
import os

files = [
    "F:/Gitlabrepo/ASQAS-Project/adc-quality-security/sheets/Scenarios_Threats.csv",
    "F:/Gitlabrepo/ASQAS-Project/adc-quality-security/sheets/Scenarios_Vulnerability.csv",
    "F:/Gitlabrepo/ASQAS-Project/adc-quality-security/src/files/Scenarios_test.csv",
]

def get_retriever(model_name):
    documents = []
    print("Loading documents...")
    for file in files:
        loader = CSVLoader(file_path=file, encoding="utf-8-sig")
        documents.extend(loader.load())
    print("Documents loaded.")

    # Split text into smaller chunks for embedding
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("Initializing embeddings...")
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model=model_name)
    print("Embeddings initialized.")

    persist_directory = "./chroma"

    # Check if Chroma DB already exists
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print("Loading existing document embeddings from ChromaDB...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Storing new document embeddings in ChromaDB...")
        try:
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
            print("Document embeddings stored.")
        except ConnectionError as e:
            print(f"Failed to connect to Ollama: {e}")
            return None

    return vectorstore.as_retriever()