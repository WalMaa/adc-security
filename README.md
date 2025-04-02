# Getting started

Requirements:
- Python
- Pip
- ollama

Install dependencies:
```bash
pip install -r requirements.txt
```

Install LLaMA 3.1 model:
```bash
ollama pull llama3.1
```

Start the Ollama server (this must be running to interact with the model):
```bash
ollama serve
```

# Environment Setup

Before running the application, you need to add your project absolute path to the `.env` file.

1. Create a `.env` file in the project root (if it doesn't exist).
2. Add the following line inside the file:
```bash
PYTHONPATH=C:/Users/yourname/path/to/adc-security
```
3. Replace the example path with the full path to your local project folder.

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