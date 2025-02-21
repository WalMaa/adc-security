import csv
import json
from llama_cpp import Llama
from rag_retriever import get_retriever
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Path to DeepSeek model file (Make sure this path is correct on Puhti)
deepseek_model_path = "src/files/DeepSeek-R1.gguf"

# Load the model
llm = Llama(model_path=deepseek_model_path, n_ctx=4096)

# Initialize the RAG retriever
model_name = "deepseek"
retriever = get_retriever(model_name)

# Initialize the ChatOllama model
chat_llm = ChatOllama(model=model_name, temperature=0.2, format="json")

# Define the prompt template
json_schema = """{
    "reasoning": "string",
    "description": "string",
    "threat_id": "string",
    "vulnerability_id": "string",
    "remediation_id": "string"
}"""

template = ChatPromptTemplate([
    ("system", "You are an assistant in security risk analysis. Analyze the following scenario and provide a reasoning, description, threat_id, vulnerability_id and remediation_id in a json format using documents:\n\n{context} \nYou need to determine if the current user message contains a security threat. \nIf a security threat is present, please explain what the security threat is. \nAnswer the following question strictly in JSON format: {json_schema}"),
    ("human", "{input}"),
])

# Create the RAG chain
combine_docs_chain = create_stuff_documents_chain(chat_llm, template)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

def read_scenarios(filename):
    """
    Read scenarios from a CSV file.
    """
    scenarios = []
    with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            scenarios.append(row)
    return scenarios

def analyze_scenarios(scenarios):
    """
    Analyze scenarios using the DeepSeek model and RAG model.
    """
    results = []
    processed_scenario_ids = set()

    for scenario in scenarios:
        scenario_id = scenario["Scenario ID"]
        if scenario_id in processed_scenario_ids:
            continue

        print(f"Analyzing scenario: {scenario_id}")

        # Create the model prompt
        prompt = f"""Analyze the following scenario and provide a reasoning, description, threat_id, vulnerability_id, and remediation_id in JSON format:
        
        Scenario: {scenario['User']}
        
        Response (JSON):
        """

        # Run DeepSeek model
        output = llm(
            prompt,
            max_tokens=300,
            temperature=0.2,
            stop=["\n\n"]
        )

        try:
            # Parse JSON output
            content = output["choices"][0]["text"].strip()
            # Clean up the output to ensure it is a valid JSON object
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                content = content[json_start:json_end]
                data = json.loads(content)

                analysis_results = {
                    "scenario_id": scenario_id,
                    "reasoning": data.get("reasoning", ""),
                    "description": data.get("description", ""),
                    "threat_id": data.get("threat_id", ""),
                    "vulnerability_id": data.get("vulnerability_id", ""),
                    "remediation_id": data.get("remediation_id", ""),
                }

                # Use RAG model to enhance the analysis
                query = scenario['User']
                try:
                    response = rag_chain.invoke({"input": query, "json_schema": json_schema})
                    enhanced_results = json.loads(response["output"])

                    # Merge the results from DeepSeek and RAG
                    analysis_results.update(enhanced_results)
                except Exception as e:
                    print(f"Error invoking RAG model for scenario {scenario_id}: {e}")

                results.append(analysis_results)
                save_to_csv(analysis_results, "analysis_results.csv")
                processed_scenario_ids.add(scenario_id)
            else:
                print(f"Invalid JSON output for scenario {scenario_id}:\n{content}")

        except json.JSONDecodeError:
            print(f"Error parsing JSON output for scenario {scenario_id}:\n{content}")

    return results

def save_to_csv(analysis_result, filename):
    """
    Save the analysis results to a CSV file.
    """
    print(f"Saving analysis results to {filename}")
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        
        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writeheader()
        
        writer.writerow(analysis_result)

def create_csv(filename):
    """
    Create a new results CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writeheader()

if __name__ == "__main__":
    create_csv("analysis_results.csv")
    scenarios = read_scenarios("src/files/Scenarios_test.csv")
    analyze_scenarios(scenarios)