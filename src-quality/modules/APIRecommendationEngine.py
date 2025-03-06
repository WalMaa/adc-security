from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class APIRecommendationEngine:
    def __init__(self, search_index):
        self.search_index = search_index
        self.api_docs = list(search_index.values())

    def search(self, query):
        """Finds the most relevant API for a given query using cosine similarity."""
        vectorizer = TfidfVectorizer()
        descriptions = [api["description"] for api in self.api_docs]
        vectors = vectorizer.fit_transform([query] + descriptions)
        similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
        best_match = self.api_docs[similarities.argmax()]
        return best_match
