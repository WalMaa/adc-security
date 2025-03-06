class APICatalogProject:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self):
        print(f"Running project: {self.name} - {self.description}")
