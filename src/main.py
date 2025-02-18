import csv
import json
from llama_cpp import Llama


deepseek_model_path = "src/files/DeepSeek-R1.gguf"

llm = Llama(model_path=deepseek_model_path, n_ctx=4096)

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
    Analyze scenarios using the DeepSeek model.
    """
    results = []

    for scenario in scenarios:
        print(f"Analyzing scenario: {scenario['Scenario ID']}")

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
                    "scenario_id": scenario["Scenario ID"],
                    "reasoning": data.get("reasoning", ""),
                    "description": data.get("description", ""),
                    "threat_id": data.get("threat_id", ""),
                    "vulnerability_id": data.get("vulnerability_id", ""),
                    "remediation_id": data.get("remediation_id", ""),
                }

                results.append(analysis_results)
                save_to_csv(analysis_results, "analysis_results.csv")
            else:
                print(f"Invalid JSON output for scenario {scenario['Scenario ID']}:\n{content}")

        except json.JSONDecodeError:
            print(f"Error parsing JSON output for scenario {scenario['Scenario ID']}:\n{content}")

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