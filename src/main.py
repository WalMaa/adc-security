import csv
from rag_implementation import prompt_llm
import json

scenario_file = "./sheets/scenarios_examples.csv"

def read_scenarios(filename):
    """
    Read scenarios from a CSV file.
    """
    scenarios = []
    with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            scenarios.append(row)
    return scenarios

def analyze_scenarios(scenarios):
    for scenario in scenarios:
        print(f"Analyzing scenario: {scenario.get('Scenario ID', 'Unknown')}")
        query = scenario.get('User', '')
        response = prompt_llm(query)
        print("Response:", response)
        try:
            res_obj = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Skipping scenario.")
            continue

        analysis_result = {
            "scenario_id": scenario.get("Scenario ID", ""),
            "reasoning": res_obj.get("reasoning", ""),
            "description": res_obj.get("description", ""),
            "threat_id": res_obj.get("threat_id", ""),
            "vulnerability_id": res_obj.get("vulnerability_id", ""),
            "remediation_id": res_obj.get("remediation_id", "")
        }
        save_to_csv(analysis_result, "analysis_results.csv")


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


create_csv("analysis_results.csv")
scenarios = read_scenarios(scenario_file)
analysis_results = analyze_scenarios(scenarios)
save_to_csv(analysis_results, "analysis_results.csv")