import csv
from rag_implementation import prompt_llm
import json
from collections import OrderedDict

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
    # Use OrderedDict instead to preserve insertion order for easy analysis
    unique_scenarios = OrderedDict()
    for scenario in scenarios:
        unique_scenarios[scenario.get('Scenario ID')] = scenario.get('User', '')
    print(f"Analyzing {len(unique_scenarios)} unique scenarios.")
    
    
    for scenario in unique_scenarios:
        scenario_id = scenario
        query = unique_scenarios[scenario]
        print(f"Analyzing scenario: {scenario_id}, Query: {query}")
        response = prompt_llm(query)
        print("Response:", response)
        try:
            res_obj = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Skipping scenario.")
            continue

        analysis_result = {
            "scenario_id": scenario_id,
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
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writerow(analysis_result)
            
def create_csv(filename):
    """
    Create a new results CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writeheader()


create_csv("analysis_results.csv")
scenarios = read_scenarios(scenario_file)
analysis_results = analyze_scenarios(scenarios)
save_to_csv(analysis_results, "analysis_results.csv")