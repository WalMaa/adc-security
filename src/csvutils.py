import csv

def save_to_csv(analysis_results, filename):
    """
    Save the analysis results to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"])
        writer.writeheader()
        for result in analysis_results:
            writer.writerow(result)