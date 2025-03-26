from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import pandas as pd
import json
import os

model_name = "llama3.1"

template = """
You are an assistant in security risk analysis.
You need to determine if the current user message contains a security threat.
If a security threat is present, please explain what the security threat is.
Find the relevant threat, vulnerability, and remediation IDs for the security threat in the provided files.
Find the relevant remediation ID in the remediation_table.csv under COUNTERMEASURE ID column, they generally start with s, pe, h or f.
Under no circumstances can you make up any id's not provided in the documents.
DO NOT HALLUCINATE.

Answer the question strictly in JSON format specified below:
{format_instructions}

Respond only with valid JSON. Do not write an introduction or summary.
Question: {query}

"""

response_schemas = [
    ResponseSchema(name="reasoning", description="Detailed reasoning about the threat."),
    ResponseSchema(name="description", description="Detailed description of the threat as described in scenarios_threats.csv."),
    ResponseSchema(name="threat_id", description="Unique identifier for the threat starting with M and found in scenarios_threats.csv."),
    ResponseSchema(name="vulnerability_id", description="Unique identifier for the associated vulnerability starting with V and found in scenarios_vulnerability.csv."),
    ResponseSchema(name="remediation_id", description="Countermeasure ID in the remediation_table.csv."),
]


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Defining a structured prompt template so that we can analyze the outputs structurally
prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
)


def preprocess_remediation_table(file_path):
    """
    Preprocess the remediation table to fill missing values and save a new copy.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.ffill(inplace=True)
        preprocessed_path = file_path.replace(".csv", "_preprocessed.csv")
        df.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")
        return preprocessed_path
    except Exception as e:
        print(f"Error processing remediation table: {e}")
        raise


def initialize_qa_chain():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        remediation_table_path = os.path.join(project_root, "sheets", "remediation_table.csv")
        threat_csv = os.path.join(project_root, "sheets", "scenarios_threats.csv")
        vulnerability_csv = os.path.join(project_root, "sheets", "scenarios_vulnerability.csv")

        preprocessed_remediation_table_path = preprocess_remediation_table(remediation_table_path)

        files = [threat_csv, vulnerability_csv, preprocessed_remediation_table_path]

        documents = []
        print("Loading documents...")
        for file in files:
            loader = CSVLoader(file_path=file, encoding="utf-8-sig")
            documents.extend(loader.load())
        print("Documents loaded.")


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
        faiss_path = os.path.join(project_root, "faiss_index_")
        vectorstore.save_local(faiss_path)
        persisted_vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        # Create a retriever
        retriever = persisted_vectorstore.as_retriever(search_kwargs={"k": 15})

        llm = ChatOllama(model=model_name, temperature=0.2,
            num_ctx=8000,
            num_predict=2048,
            format="json",
            )

        # Create Retrieval-Augmented Generation (RAG) system
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff" , retriever=retriever)
    except Exception as e:
        print(f"Initialization failed: {e}")
        raise


def prompt_llm(query):
    """
    Formats and sends a query to the LLM via RetrievalQA.
    """
    formatted_prompt = prompt.format(query=query, format_instructions=format_instructions)
    print("Querying:", query)

    try:
        qa_chain = initialize_qa_chain()
        raw_response = qa_chain(formatted_prompt)

        if isinstance(raw_response, dict):
            if "result" in raw_response:
                try:
                    return json.dumps(json.loads(raw_response["result"]))
                except json.JSONDecodeError:
                    raise ValueError("Invalid nested JSON in 'result' field.")
            return json.dumps(raw_response)

        json.loads(raw_response)
        return raw_response

    except ValueError:
        raise
    except Exception as e:
        print(f"Error during prompt invocation: {e}")
        return json.dumps({
            "reasoning": "",
            "description": "",
            "threat_id": "",
            "vulnerability_id": "",
            "remediation_id": ""
        })

