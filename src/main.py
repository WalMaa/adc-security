import os
import csv
import json
from collections import OrderedDict
from src.rag_implementation import prompt_llm, initialize_qa_chain

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

scenario_file = os.path.join(project_root, "sheets", "scenarios_examples.csv")
analysis_results_file = os.path.join(project_root, "analysis_results.csv")


def read_scenarios(filename):
    """
    Read scenarios from a CSV file.
    """
    scenario_list = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                scenario_list.append(row)
    except FileNotFoundError:
        print(f"Scenario file not found: {filename}")
    return scenario_list


def analyze_scenarios(scenario_list, qa_chain):
    """
    Analyze each unique scenario and save the results.
    """
    unique_scenarios = OrderedDict()
    for scenario in scenario_list:
        unique_scenarios[scenario.get('Scenario ID')] = scenario.get('User', '')
    print(f"Analyzing {len(unique_scenarios)} unique scenarios.")

    for scenario_id, query in unique_scenarios.items():
        print(f"Analyzing scenario: {scenario_id}, Query: {query}")
        response = prompt_llm(query, qa_chain)
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
        save_to_csv(analysis_result, analysis_results_file)


def save_to_csv(analysis_result, filename):
    """
    Save the analysis results to a CSV file.
    """
    print(f"Saving analysis results to {filename}")
    try:
        with open(filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id",
                                                      "vulnerability_id", "remediation_id"])
            writer.writerow(analysis_result)
    except Exception as e:
        print(f"Error saving analysis results: {e}")


def create_csv(filename):
    """
    Create a new results CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file,
                                fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id",
                                            "remediation_id"])
        writer.writeheader()


def main():
    create_csv(analysis_results_file)
    scenarios = read_scenarios(scenario_file)
    qa_chain = initialize_qa_chain()
    analyze_scenarios(scenarios, qa_chain)


if __name__ == "__main__": # pragma: no cover
    main()
