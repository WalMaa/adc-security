import requests
import csv
import json


scenario_file = "C:/Code/Python/advanced_quality_and_security/src/files/Scenarios_test.csv"

# Define the base URL for the LM Studio API
base_url = "http://127.0.0.1:1234"

# Endpoint to get the list of available models
models_endpoint = "/v1/models"

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

def analyze_scenarios(scenarios, model_id):
    
    for scenario in scenarios:
        print(f"Analyzing scenario: {scenario['Scenario ID']}")
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Analyze the following scenario and provide a reasoning, description, threat_id, vulnerability_id and remediation_id in a json format:\n\n{scenario['User']}"}
                        ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "risk_analysis",
                            "strict": "true",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reasoning": {"type": "string"},
                                    "description": {"type": "string"},
                                    "threat_id": {"type": "string"},
                                    "vulnerability_id": {"type": "string"},
                                    "remediation_id": {"type": "string"}
                                },
                                "required": ["reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"]
                            }
                        },
                    },
                    "max_tokens": 300,
                    # Low temperature to reduce randomness in the response
                    "temperature": 0.2,
                }
            )

            if response.status_code == 200:
                result = response.json()

                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                data = json.loads(content)
                
                reasoning = data["reasoning"]
                description = data["description"]
                # for example M1
                threat_id = data["threat_id"]
                # for example V1
                vulnerability_id = data["vulnerability_id"]
                remediation_id = data["remediation_id"]
                
                analysis_results = {
                    "scenario_id": scenario["Scenario ID"],
                    "reasoning": reasoning,
                    "description": description,
                    "threat_id": threat_id,
                    "vulnerability_id": vulnerability_id,
                    "remediation_id": remediation_id,
                }
                print(analysis_results)
                save_to_csv(analysis_results, "analysis_results.csv")

            else:
                print(f"Failed to analyze scenario. Status code: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"An error occurred: {e}")


def save_to_csv(analysis_result, filename):
    """
    Save the analysis results to a CSV file.
    """
    print(f"Saving analysis results to {filename}")
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writerow(analysis_result)
            
def create_csv(filename):
    """
    Create a new results CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writeheader()


scenarios = read_scenarios(scenario_file)
analysis_results = analyze_scenarios(scenarios, "deepseek-r1-distill-qwen-7b")
save_to_csv(analysis_results, "analysis_results.csv")