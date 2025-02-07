import requests
import csv
import time
import json

# Define the base URL for the LM Studio API
base_url = "http://127.0.0.1:1234"

# Endpoint to get the list of available models
models_endpoint = "/v1/models"

def get_models():
    """
    Fetch the list of available models from LM Studio.
    """
    try:
        response = requests.get(base_url + models_endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch models. Status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"An error occurred: {e}")

def read_scenarios(filename):
    """
    Read scenarios from a CSV file.
    """
    scenarios = []
    with open(filename, mode='r', newline='', encoding='ISO-8859-1') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            scenarios.append(row)
    return scenarios

def analyze_scenarios(scenarios, model_id):
    analysis_results = []
    for scenario in scenarios:
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": f"Analyze the following scenario and provide a summary, risks, and mitigations:\n\n{scenario['Assistant - Extended']}"}],
                    "max_tokens": 150
                }
            )

            if response.status_code == 200:
                result = response.json()
                print("Response:", result)  # Debugging output

                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                response_time = time.time() - start_time

                # Parse the content to extract summary, risks, and mitigations
                # This is a placeholder; you may need to adjust the parsing logic based on the actual response format
                summary = content.split("\n")[0] if content else "No summary provided"
                risks = ["Risk 1", "Risk 2"]  # Placeholder, adjust based on actual response
                mitigations = ["Mitigation 1", "Mitigation 2"]  # Placeholder, adjust based on actual response

                analysis_results.append({
                    "summary": summary,
                    "risks": json.dumps(risks),
                    "mitigations": json.dumps(mitigations),
                    "response_time": response_time,
                    "scenario_id": scenario["Scenario ID"],
                    "scenario": scenario["Assistant - Extended"]
                })

                time.sleep(1)  # Avoid spamming API calls
            else:
                print(f"Failed to analyze scenario. Status code: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"An error occurred: {e}")

    return analysis_results

def save_to_csv(analysis_results, filename):
    """
    Save the analysis results to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["summary", "risks", "mitigations", "response_time", "scenario_id", "scenario"])
        writer.writeheader()
        for result in analysis_results:
            writer.writerow(result)

# Example usage
models = get_models()
print(models)

scenarios = read_scenarios('F:/Gitlabrepo/ASQAS-Project/adc-quality-security/src/files/Scenarios_test.csv')
analysis_results = analyze_scenarios(scenarios, "deepseek-r1-distill-qwen-7b")
save_to_csv(analysis_results, "analysis_results.csv")