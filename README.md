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

# Environment Setup

Before running the application, you need add your project absolute path to the `.env` file.

For example:
```bash
PYTHONPATH=C:/Users/yourname/path/to/adc-security
```

# Run the Analysis

Run `main.py` to start analysis:
```bash
python main.py
```
# Testing

Run tests in project root:
```bash
pytest src/tests --cov=src --cov-report=term-missing --tb=short -v
```