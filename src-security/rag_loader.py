from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
import os
from langchain_ollama import OllamaEmbeddings


from langchain_text_splitters import RecursiveCharacterTextSplitter


persist_directory = "./chroma"

files = [
"./sheets/scenarios_threats.csv",
"./sheets/scenarios_vulnerability.csv",
"./sheets/scenarios_examples.csv",
]


def load_rag(model_name):
    
    documents = []
    print("Loading documents...")
    for file in files:
        loader = CSVLoader(file_path=file, encoding="utf-8-sig")
        documents.extend(loader.load())
    print("Documents loaded.")
    
    # Split text into smaller chunks for embedding
    csv_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Adjust based on your desired maximum chunk length
    chunk_overlap=50,        # Overlap between chunks if needed
    separators=["\n", ",", " "]  # Prioritize newlines, then commas, then whitespace
    )

    docs = csv_splitter.split_documents(documents)


    print("Initializing embeddings...")
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model=model_name)
    print("Embeddings initialized.")
    
    # Check if Chroma DB already exists
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print("Loading existing document embeddings from ChromaDB...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Storing new document embeddings in ChromaDB...")
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        print("Document embeddings stored.")
        
    # Load retriever
    return vectorstore.as_retriever(search_kwargs={"k": 50})
    