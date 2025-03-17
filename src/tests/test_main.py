import pytest
import csv
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.main import read_scenarios, analyze_scenarios, save_to_csv, create_csv

@pytest.fixture
def sample_scenarios():
    """
    Fixture providing a sample scenario for testing.

    Returns:
        list: A list containing a single dictionary representing a scenario.
    """
    return [{"Scenario ID": "1", "User": "Test Query"}]

def test_read_scenarios(monkeypatch, tmp_path):
    """
    Tests the `read_scenarios` function to ensure it correctly reads
    scenarios from a CSV file

    Steps:
    - Creates a temporary CSV file with sample scenario data.
    - Calls `read_scenarios` to read the data.
    - Asserts that the read data matches the expected structure

    Args:
        monkeypatch (pytest fixture): Used to modify behavior of system functions.
        tmp_path (pytest fixture): Provides a temporary directory for file operations.
    """
    csv_file = tmp_path / "test_scenarios.csv"
    with open(csv_file, "w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario ID", "User"])
        writer.writerow(["1", "Test Query"])

    scenarios = read_scenarios(csv_file)
    assert len(scenarios) == 1
    assert scenarios[0]["Scenario ID"] == "1"
    assert scenarios[0]["User"] == "Test Query"

def test_analyze_scenarios(mocker, sample_scenarios):
    """
    Tests the `analyze_scenarios` function to ensure it calls the
    `prompt_llm` function with the correct query.

    Steps:
    - Mocks the `prompt_llm` function to return a predefined JSON response.
    - Calls `analyze_scenarios` with a sample scenario.
    - Asserts that `prompt_llm` was called once with the correct query.

    Args:
        mocker (pytest fixture): Used to patch and mock functions.
        sample_scenarios (pytest fixture): Provides a sample scenario for testing.
    """
    mock_prompt = mocker.patch("src.main.prompt_llm", return_value='{"reasoning": "test", '
                                                                   '"description": "test", '
                                                                   '"threat_id": "M1", '
                                                                   '"vulnerability_id": "V1", '
                                                                   '"remediation_id": "s1"}')
    analyze_scenarios(sample_scenarios)
    mock_prompt.assert_called_once_with("Test Query")

def test_save_to_csv(tmp_path):
    """
    Tests the `save_to_csv` function to ensure that analysis results
    are correctly written to a CSV file.

    Steps:
    - Creates a temporary file for CSV output.
    - Calls `save_to_csv` with sample analysis data.
    - Reads the file and verifies the written content.

    Args:
        tmp_path (pytest fixture): Provides a temporary directory for file operations.
    """
    csv_file = tmp_path / "test_analysis.csv"
    data = {"scenario_id": "1",
            "reasoning": "Test Reasoning",
            "description": "Test Desc",
            "threat_id": "M1",
            "vulnerability_id": "V1",
            "remediation_id": "s1"}

    save_to_csv(data, csv_file)

    with open(csv_file, "r", encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "1"
    assert rows[0]["reasoning"] == "Test Reasoning"

def test_create_csv(tmp_path):
    """
    Tests the `create_csv` function to ensure that a CSV file
    is created with the correct header structure

    Steps:
    - Calls `create_csv` to create a new CSV file.
    - Reads the file and verifies that the headers match the expected format

    Args:
        tmp_path (pytest fixture): Provides a temporary directory for file operations.
    """
    csv_file = tmp_path / "test_new_results.csv"
    create_csv(csv_file)

    with open(csv_file, "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)

    assert headers == ["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"]
