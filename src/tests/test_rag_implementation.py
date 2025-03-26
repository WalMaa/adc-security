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
    """
    csv_file = tmp_path / "test_remediation.csv"
    df = pd.DataFrame({"COUNTERMEASURE ID": ["s1", "pe2", "h3", "f4"]})
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    return csv_file


def test_preprocess_remediation_table(sample_csv):
    """
    Tests the `preprocess_remediation_table` function to ensure it correctly preprocesses
    and saves the remediation table.
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

def test_initialize_qa_chain_success(mocker):
    """
    Tests that initialize_qa_chain completes successfully and returns a RetrievalQA object.
    All dependencies are mocked to isolate logic.
    """
    from src import rag_implementation

    # Patch preprocess_remediation_table
    mocker.patch("src.rag_implementation.preprocess_remediation_table", return_value="./fake.csv")

    # Patch CSVLoader
    mock_loader = mocker.patch("src.rag_implementation.CSVLoader")
    mock_loader.return_value.load.return_value = [{"page_content": "test doc"}]

    # Patch CharacterTextSplitter
    mock_splitter = mocker.patch("src.rag_implementation.CharacterTextSplitter")
    mock_splitter.return_value.split_documents.return_value = ["chunk1", "chunk2"]

    # Patch HuggingFaceEmbeddings
    mock_embedding_instance = mocker.Mock()
    mock_embed = mocker.patch("src.rag_implementation.HuggingFaceEmbeddings", return_value=mock_embedding_instance)

    # Patch FAISS
    mock_vectorstore_instance = mocker.Mock()
    mock_faiss = mocker.patch("src.rag_implementation.FAISS")
    mock_faiss.from_documents.return_value = mock_vectorstore_instance
    mock_vectorstore_instance.save_local.return_value = None

    mock_persisted_store = mocker.Mock()
    mock_persisted_store.as_retriever.return_value = "mock_retriever"
    mock_faiss.load_local.return_value = mock_persisted_store

    # Patch ChatOllama
    mock_llm_instance = mocker.Mock()
    mock_llm = mocker.patch("src.rag_implementation.ChatOllama", return_value=mock_llm_instance)

    # Patch RetrievalQA
    mock_qa = mocker.patch("src.rag_implementation.RetrievalQA")
    mock_qa.from_chain_type.return_value = "mock_qa_chain"

    # Run the function
    result = rag_implementation.initialize_qa_chain()

    # Assertion
    assert result == "mock_qa_chain"

    # Verify calls
    mock_embed.assert_called_once_with(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda"}
    )
    mock_llm.assert_called_once_with(
        model="llama3.1",
        temperature=0.2,
        num_ctx=8000,
        num_predict=2048,
        format="json"
    )
    mock_faiss.from_documents.assert_called_once_with(["chunk1", "chunk2"], mock_embedding_instance)

    # Match full faiss_path, not just "faiss_index_"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    faiss_path = os.path.join(project_root, "faiss_index_")

    mock_faiss.load_local.assert_called_once_with(
        faiss_path,
        mock_embedding_instance,
        allow_dangerous_deserialization=True
    )

    mock_qa.from_chain_type.assert_called_once_with(
        llm=mock_llm_instance,
        chain_type="stuff",
        retriever="mock_retriever"
    )


def test_initialize_qa_chain_failure(monkeypatch, capsys):
    """
    Tests that initialize_qa_chain handles and prints initialization errors.
    """
    from src import rag_implementation

    monkeypatch.setattr(rag_implementation, "preprocess_remediation_table",
                        lambda _: (_ for _ in ()).throw(RuntimeError("Mocked init failure")))

    with pytest.raises(RuntimeError):
        rag_implementation.initialize_qa_chain()

    captured = capsys.readouterr()
    assert "Initialization failed: Mocked init failure" in captured.out


def test_prompt_llm(mocker):
    """
    Tests that prompt_llm returns a valid JSON object when QA chain works correctly.
    """
    mock_chain = mocker.Mock()
    mock_chain.return_value = ('{"reasoning": "Test", '
                               '"description": "Test Threat", '
                               '"threat_id": "M1", '
                               '"vulnerability_id": "V1", '
                               '"remediation_id": "s1"}')

    mocker.patch("src.rag_implementation.initialize_qa_chain", return_value=mock_chain)
    response = prompt_llm("Test Query")
    assert isinstance(response, str)

    parsed = json.loads(response)
    assert parsed["reasoning"] == "Test"
    assert parsed["description"] == "Test Threat"
    assert parsed["threat_id"] == "M1"
    assert parsed["vulnerability_id"] == "V1"
    assert parsed["remediation_id"] == "s1"


def test_prompt_llm_json_error(mocker):
    """
    Tests that prompt_llm handles malformed JSON gracefully.
    """
    mock_chain = mocker.Mock()
    mock_chain.return_value = "{invalid json"

    mocker.patch("src.rag_implementation.initialize_qa_chain", return_value=mock_chain)

    response = prompt_llm("Some query")
    parsed = json.loads(response)

    assert parsed["reasoning"] == ""
    assert parsed["description"] == ""
    assert parsed["threat_id"] == ""
    assert parsed["vulnerability_id"] == ""
    assert parsed["remediation_id"] == ""
