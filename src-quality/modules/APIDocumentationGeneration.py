import json
from langchain_ollama import ChatOllama


class APIDocumentationGeneration:
    def __init__(self, api_endpoints, model="llama3.1"):
        self.api_endpoints = api_endpoints
        self.llm_model = ChatOllama(model=model, format="json", temperature=0.5)

    def generate_documentation(self):
        """Generates documentation for extracted API endpoints using Ollama."""
        api_docs = []
        for api in self.api_endpoints:
            prompt = f"Document the API endpoint: {api['route']} (Method: {api['method']}). Function: {api['function']}."

            response = self.llm_model.invoke(prompt)
            response_json = response.model_dump_json()

            try:
                response_dict = json.loads(response_json)
                description = response_dict.get("response", "Failed to generate description.")
            except json.JSONDecodeError:
                description = "Failed to parse response."

            api_docs.append({
                "route": api["route"],
                "method": api["method"],
                "description": description
            })
        return api_docs

    def format_documentation(self):
        """Formats the generated API documentation."""
        return f"API Documentation: {len(self.api_endpoints)} endpoints documented."
