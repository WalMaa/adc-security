import ast

class CodeParsingAndAPIExtraction:
    def __init__(self, codebase_path):
        self.codebase_path = codebase_path
        self.api_endpoints = []

    def parse_codebase(self):
        with open(self.codebase_path, "r", encoding="utf-8") as file:
            return file.read()

    def extract_apis(self):
        tree = ast.parse(self.parse_codebase())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr'):
                        if decorator.func.attr in ["route", "get", "post", "put", "delete"]:
                            self.api_endpoints.append({
                                "function": node.name,
                                "route": decorator.args[0].s if decorator.args else None,
                                "method": decorator.func.attr
                            })
        return self.api_endpoints
