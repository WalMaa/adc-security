from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from rag_loader import load_rag

model_name = "mistral"

response_schemas = [
    ResponseSchema(name="reasoning", description="Detailed reasoning about the threat."),
    ResponseSchema(name="description", description="Detailed description of the threat as described in scenarios_threats.csv."),
    ResponseSchema(name="threat_id", description="Unique identifier for the threat starting with M and found in scenarios_threats.csv."),
    ResponseSchema(name="vulnerability_id", description="Unique identifier for the associated vulnerability starting with V and found in scenarios_vulnerability.csv."),
    ResponseSchema(name="remediation_id", description="Countermeasure ID in the remediation_table.csv."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

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

Question: {query}

"""

# Defining a structured prompt template so that we can analyze the outputs structurally
prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
)

retriever = load_rag(model_name)

# Load DeepSeek-R1 model via Ollama
llm = ChatOllama(model=model_name, format="json", temperature=0.5)

# Create Retrieval-Augmented Generation (RAG) system
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def prompt_llm(query):
    formatted_prompt = prompt.format(query=query, format_instructions=format_instructions)
    print("Querying:", query)
    response = qa_chain.run(formatted_prompt)
    return response