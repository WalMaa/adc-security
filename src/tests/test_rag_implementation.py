import json
import pytest
import sys
import os
import pandas as pd
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag_implementation import preprocess_remediation_table, prompt_llm


@pytest.fixture
def sample_csv(tmp_path):
    """
    Fixture that creates a temporary CSV file containing sample countermeasure data.

    Returns:
        pathlib.Path: The path to the temporary CSV file.

    Args:
        tmp_path (pytest fixture): Provides a temporary directory for file operations.
    """
    csv_file = tmp_path / "test_remediation.csv"
    df = pd.DataFrame({"COUNTERMEASURE ID": ["s1", "pe2", "h3", "f4"]})
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    return csv_file


def test_preprocess_remediation_table(sample_csv):
    """
    Tests the `preprocess_remediation_table` function to ensure it correctly preprocesses
    and saves the remediation table.

    Steps:
    - Calls `preprocess_remediation_table` with a sample CSV file.
    - Verifies that the preprocessed file is created.
    - Reads the file to ensure the structure is preserved.

    Args:
        sample_csv (pytest fixture): Provides a sample CSV file for testing.
    """
    new_path = preprocess_remediation_table(str(sample_csv))
    assert os.path.exists(new_path)
    df = pd.read_csv(new_path, encoding="utf-8-sig")
    assert "COUNTERMEASURE ID" in df.columns
    assert len(df) == 4


def test_preprocess_remediation_table_file_not_found():
    """
    Tests that FileNotFoundError is raised when a nonexistent file is passed.
    """
    with pytest.raises(FileNotFoundError):
        preprocess_remediation_table("nonexistent_file.csv")


def test_prompt_llm(mocker):
    """
    Tests the `prompt_llm` function to ensure it correctly calls the QA model
    and returns a structured JSON response.

    Steps:
    - Mocks the `qa_chain` method to return a predefined JSON response.
    - Calls `prompt_llm` with a test query.
    - Asserts that `qa_chain` was called exactly once.
    - Verifies that the returned response contains all expected fields.

    Args:
        mocker (pytest fixture): Used to mock the `qa_chain` function.
    """
    mock_qa_chain = mocker.patch("src.rag_implementation.qa_chain",
                                 return_value='{"reasoning": "Test", '
                                              '"description": "Test Threat", '
                                              '"threat_id": "M1", '
                                              '"vulnerability_id": "V1", '
                                              '"remediation_id": "s1"}')
    response = prompt_llm("Test Query")
    mock_qa_chain.assert_called_once()
    assert "reasoning" in response
    assert "description" in response
    assert "threat_id" in response
    assert "vulnerability_id" in response
    assert "remediation_id" in response


def test_prompt_llm_json_error(monkeypatch):
    """
    Tests that prompt_llm handles malformed JSON gracefully.
    """
    with patch("src.rag_implementation.qa_chain", return_value="{invalid json"):
        response = prompt_llm("Test query")
        parsed = json.loads(response)
        assert parsed["reasoning"] == ""
        assert parsed["description"] == ""
        assert parsed["threat_id"] == ""
        assert parsed["vulnerability_id"] == ""
        assert parsed["remediation_id"] == ""
