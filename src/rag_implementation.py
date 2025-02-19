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

class Scenario(BaseModel):
    reasoning: str
    description: str
    threat_id: str
    vulnerability_id: str
    remediation_id: str

model_name = "mistral"
files = [
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_threats.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_vulnerability.csv",
    "C:/Code/Python/advanced_quality_and_security/sheets/scenarios_examples.csv",
]



template = """
Analyze the following scenario and provide reasoning, description, threat_id, vulnerability_id, remediation_id in JSON. If an appropriate value is not available, please set it as null but always include the keys.
Context:
{context}
Scenario:
{input}
"""


# Defining a structured prompt template so that we can analyze the outputs structurally
prompt = PromptTemplate(
    template=template,
    input_variables=["input"],
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
llm = ChatOllama(model=model_name, temperature=0.4, num_predict=1000)
structured_llm = llm.with_structured_output(Scenario)

# Create Retrieval-Augmented Generation (RAG) system
print("Initializing QA chain...")
question_answer_chain = create_stuff_documents_chain(structured_llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
print("QA chain initialized.")

query = "I have a case where my server room is in a basement and we have structural damage in the building."
print("Querying:", query)
response = chain.invoke({"input": query})

print("Response:", response)