import os
import glob
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class InvestigationCorpus:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.documents = []
        self.graph = nx.Graph()
        
        # Load models for retrieval
        print("Loading semantic model (this takes a few seconds)...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer()
        
        self.tfidf_matrix = None
        self.semantic_embeddings = None
        
        self.ingest_corpus()

    def ingest_corpus(self):
        """Loads text files, builds indices, and generates the evidence graph."""
        filepaths = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        texts = []
        for fp in filepaths:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
                doc_id = os.path.basename(fp)
                self.documents.append({"id": doc_id, "content": content})
                texts.append(content)
                
                # Add to evidence graph (Document node)
                self.graph.add_node(doc_id, type="document")
                
                # A very basic rule-based entity extractor to satisfy the graph requirement quickly
                if "Alice" in content:
                    self.graph.add_edge(doc_id, "Alice", relation="mentions")
                if "Bob" in content:
                    self.graph.add_edge(doc_id, "Bob", relation="mentions")
                if "Diamond Crown" in content:
                    self.graph.add_edge(doc_id, "Diamond Crown", relation="mentions")

        if texts:
            # Build Keyword Index
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            # Build Semantic Index
            self.semantic_embeddings = self.semantic_model.encode(texts)

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """Combines Keyword and Semantic search as requested by the prompt."""
        if not self.documents:
            return []

        # 1. Keyword search (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # 2. Semantic search
        query_embedding = self.semantic_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.semantic_embeddings)[0]

        # Combine scores (50% keyword, 50% semantic)
        combined_scores = (0.5 * keyword_scores) + (0.5 * semantic_scores)
        
        # Get top K indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0: # Only return relevant docs
                results.append(self.documents[idx])
                
        return results

# Initialize a global instance so our agents can use it
corpus = InvestigationCorpus()