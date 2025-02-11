from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import os

model_name = "deepseek-r1:7b"
files = [
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_threats.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_vulnerability.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_examples.csv",
    
]

response_schemas = [
    ResponseSchema(name="reasoning", description="Detailed reasoning about the threat."),
    ResponseSchema(name="description", description="Detailed description of the threat."),
    ResponseSchema(name="threat_id", description="Unique identifier for the threat."),
    ResponseSchema(name="vulnerability_id", description="Unique identifier for the associated vulnerability."),
    ResponseSchema(name="remediation_id", description="Unique identifier for the remediation."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Defining a structured prompt template so that we can analyze the outputs structurally
prompt = PromptTemplate(
    template="Answer the following question strictly in JSON format:\n\n{format_instructions}\n\nQuestion: {query}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)


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
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    print("Document embeddings stored.")

retriever = vectorstore.as_retriever()

# Load DeepSeek-R1 model via Ollama
llm = OllamaLLM(model=model_name)

# Create Retrieval-Augmented Generation (RAG) system
print("Initializing QA chain...")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("QA chain initialized.")

query = "What is the threat and description of threat id M1"
formatted_prompt = prompt.format(query=query)
print("Querying:", query)
response = qa_chain.run(query)

print("Response:", response)