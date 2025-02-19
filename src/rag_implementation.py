from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

model_name = "mistral"
files = [
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_threats.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_vulnerability.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_examples.csv",
]



template = """
You are an expert analyst. Using the context provided below only as background information, analyze the following scenario and produce a new JSON output with the following keys: 
- reasoning
- description: a description that corresponds with the threat ID
- threat_id: starts with "M" and is followed by a number
- vulnerability_id: starts with "V" and is followed by a number
- remediation_id: Assign a unique identifier to the remediation.

Do not simply return or repeat the context. If a value is not applicable, set it to null.

Get the applicable ids from the following lists:
{context}

Scenario:
{question}
"""

query = "What is the description of threat id m2"


# Defining a structured prompt template so that we can analyze the outputs structurally
prompt_template = PromptTemplate.from_template(template)

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
    

# Create a retrieval chain
retriever = vectorstore.as_retriever()
relevant_docs = retriever.get_relevant_documents(query)
context_text = "\n".join([doc.page_content for doc in relevant_docs])



prompt = prompt_template.format(context=context_text, question=query)

# Load DeepSeek-R1 model via Ollama
llm = ChatOllama(model=model_name, temperature=0.4, format="json")

response_text = llm.invoke(prompt)

print(response_text)