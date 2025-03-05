from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from rag_loader import load_rag
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

files = [
"./sheets/scenarios_threats.csv",
"./sheets/scenarios_vulnerability.csv",
# "./sheets/scenarios_examples.csv",
"./sheets/remediation_table.csv"
]

documents = []
print("Loading documents...")
for file in files:
    loader = CSVLoader(file_path=file, encoding="utf-8-sig")
    documents.extend(loader.load())
print("Documents loaded.")

model_name = "llama3.1"

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
docs = text_splitter.split_documents(documents=documents)


# Load embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save and reload the vector store
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = persisted_vectorstore.as_retriever(search_kwargs={"k": 15})
# retriever = load_rag(model_name=model_name)

llm = ChatOllama(model=model_name, temperature=0.2,
    num_ctx=8000,
    num_predict=2048)

# Create Retrieval-Augmented Generation (RAG) system
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff" , retriever=retriever)

result = qa_chain.invoke("I have flooding in my server room, give a proper threat id, vulnerability id and countermeasure id for this",)
print(result)

