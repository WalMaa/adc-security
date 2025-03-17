# Getting started

Requirements:
- Python
- Pip
- ollama

Install mistral using the following command:

```bash
ollama pull llama3.1
```

Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

Run main.py to start analysis

# Testing

Run tests in project root:
```bash
pytest src/tests --cov=src --cov-report=term-missing
```