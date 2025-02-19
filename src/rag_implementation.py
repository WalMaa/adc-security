
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from rag_retriever import get_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

    
model_name = "mistral"

json_schema = """{
    "reasoning": "string",
    "description": "string",
    "threat_id": "string",
    "vulnerability_id": "string",
    "remediation_id": "string",
}"""

template = ChatPromptTemplate([
    ("system", "You are an assistant in security risk analysis. Analyze the following scenario and provide a reasoning, description, threat_id, vulnerability_id and remediation_id in a json format using documents:\n\n{context} \nYou need to determine if the current user message contains a security threat. \nIf a security threat is present, please explain what the security threat is. \nAnswer the following question strictly in JSON format: {json_schema}"),
    ("human", "{input}"),
])


retriever = get_retriever(model_name)

llm = ChatOllama(model=model_name, temperature=0.2, format="json")

query = "I have a case where my server room is in a basement and we have structural damage in the building."
# Create Retrieval-Augmented Generation (RAG) system

combine_docs_chain = create_stuff_documents_chain(llm, template )
raq_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoke the RAG system
response = raq_chain.invoke({"input": query, "json_schema": json_schema})

print(response)
