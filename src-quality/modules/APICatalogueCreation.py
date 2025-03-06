import json
import sqlite3

class APICatalogueCreation:
    def __init__(self, api_documents):
        self.api_documents = api_documents
        self.search_index = {}

    def create_catalogue(self, filename="api_catalogue.json"):
        """Saves API documentation to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.api_documents, f, indent=4)

    def build_search_index(self):
        """Creates an index for fast API lookups."""
        self.search_index = {doc["route"]: doc for doc in self.api_documents}
        return self.search_index

    def save_to_db(self, db_name="api_catalogue.db"):
        """Stores API documentation in an SQLite database."""
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS api_catalogue (route TEXT, method TEXT, description TEXT)")
        for api in self.api_documents:
            cursor.execute("INSERT INTO api_catalogue VALUES (?, ?, ?)",
                           (api["route"], api["method"], api["description"]))
        conn.commit()
        conn.close()
