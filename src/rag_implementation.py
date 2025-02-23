from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from rag_loader import load_rag

model_name = "deepseek-r1:7b"

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
    template="Answer the following question strictly in JSON format:\n\n{format_instructions}\n\nQuestion: {query}",
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