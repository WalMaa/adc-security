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
    """
    return [{"Scenario ID": "1", "User": "Test Query"}]


def test_read_scenarios(monkeypatch, tmp_path):
    """
    Tests the `read_scenarios` function to ensure it correctly reads
    scenarios from a CSV file
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


def test_read_scenarios_file_not_found():
    """
    Tests that read_scenarios handles missing file gracefully.
    """
    scenarios = read_scenarios("nonexistent_file.csv")
    assert scenarios == []


def test_analyze_scenarios(mocker, sample_scenarios):
    """
    Tests the `analyze_scenarios` function to ensure it calls the
    `prompt_llm` function with the correct query.
    """
    mock_prompt = mocker.patch("src.main.prompt_llm", return_value='{"reasoning": "test", '
                                                                   '"description": "test", '
                                                                   '"threat_id": "M1", '
                                                                   '"vulnerability_id": "V1", '
                                                                   '"remediation_id": "s1"}')
    analyze_scenarios(sample_scenarios)
    mock_prompt.assert_called_once_with("Test Query")


def test_analyze_scenarios_invalid_json(mocker, sample_scenarios):
    """
    Tests that analyze_scenarios handles JSON decoding errors gracefully.
    """
    mock_prompt = mocker.patch("src.main.prompt_llm", return_value="{invalid json")
    result = analyze_scenarios(sample_scenarios)
    assert result is None
    mock_prompt.assert_called_once()


def test_save_to_csv(tmp_path):
    """
    Tests the `save_to_csv` function to ensure that analysis results
    are correctly written to a CSV file.
    """
    csv_file = tmp_path / "test_analysis.csv"
    create_csv(csv_file)

    data = {
        "scenario_id": "1",
        "reasoning": "Test Reasoning",
        "description": "Test Desc",
        "threat_id": "M1",
        "vulnerability_id": "V1",
        "remediation_id": "s1"
    }

    save_to_csv(data, csv_file)

    with open(csv_file, "r", encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "1"


def test_save_to_csv_exception_handling(tmp_path, capsys, mocker):
    """
    Tests that save_to_csv handles exceptions gracefully when file writing fails.
    """
    data = {
        "scenario_id": "1",
        "reasoning": "Test Reasoning",
        "description": "Test Desc",
        "threat_id": "M1",
        "vulnerability_id": "V1",
        "remediation_id": "s1"
    }

    mocker.patch("builtins.open", side_effect=IOError("Mocked write error"))
    save_to_csv(data, tmp_path / "failing.csv")
    captured = capsys.readouterr()
    assert "Error saving analysis results: Mocked write error" in captured.out


def test_create_csv(tmp_path):
    """
    Tests the `create_csv` function to ensure that a CSV file
    is created with the correct header structure
    """
    csv_file = tmp_path / "test_new_results.csv"
    create_csv(csv_file)

    with open(csv_file, "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)

    assert headers == ["scenario_id", "reasoning", "description", "threat_id", "vulnerability_id", "remediation_id"]


def test_main_flow(mocker, tmp_path):
    """
    Tests the main flow by mocking all subcomponents to ensure the pipeline runs without error.
    """
    # Create temporary file paths
    dummy_scenario_path = tmp_path / "scenarios.csv"
    dummy_result_path = tmp_path / "results.csv"

    # Mock the file variables in the main module
    mocker.patch("src.main.scenario_file", str(dummy_scenario_path))
    mocker.patch("src.main.analysis_results_file", str(dummy_result_path))

    # Mock the actual logic functions
    mock_create = mocker.patch("src.main.create_csv")
    mock_read = mocker.patch("src.main.read_scenarios", return_value=[{"Scenario ID": "1",
                                                           "User": "test"}])
    mock_analyze = mocker.patch("src.main.analyze_scenarios", return_value={"scenario_id": "1",
                                                             "reasoning": "r",
                                                             "description": "d",
                                                             "threat_id": "M1",
                                                             "vulnerability_id": "V1",
                                                             "remediation_id": "s1"})

    mock_save = mocker.patch("src.main.save_to_csv")

    # Run main()
    from src.main import main
    main()

    mock_create.assert_called_once_with(str(dummy_result_path))
    mock_read.assert_called_once_with(str(dummy_scenario_path))
    mock_analyze.assert_called_once()
    mock_save.assert_called_once()
