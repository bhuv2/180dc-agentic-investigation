import os
import glob
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Ensure the Gemini API key is configured
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class InvestigationCorpus:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.documents = []
        self.graph = nx.Graph()
        
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.semantic_embeddings = []
        
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
                
                # Basic entity extraction
                if "Alice" in content:
                    self.graph.add_edge(doc_id, "Alice", relation="mentions")
                if "Bob" in content:
                    self.graph.add_edge(doc_id, "Bob", relation="mentions")
                if "Diamond Crown" in content:
                    self.graph.add_edge(doc_id, "Diamond Crown", relation="mentions")

        if texts:
            # 1. Keyword Index (TF-IDF)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # 2. Semantic Index via Gemini API (Lightweight, No RAM overhead)
            print("Fetching Gemini embeddings for corpus...")
            for text in texts:
                emb = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                self.semantic_embeddings.append(emb['embedding'])
            self.semantic_embeddings = np.array(self.semantic_embeddings)

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """Combines Keyword and Semantic search."""
        if not self.documents:
            return []

        # 1. Keyword search (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # 2. Semantic search (Gemini API)
        query_emb = genai.embed_content(
            model="models/text-embedding-004",
            content=query
        )['embedding']
        semantic_scores = cosine_similarity([query_emb], self.semantic_embeddings)[0]

        # Combine scores (50% keyword, 50% semantic)
        combined_scores = (0.5 * keyword_scores) + (0.5 * semantic_scores)
        
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                results.append(self.documents[idx])
                
        return results

# Initialize global instance
corpus = InvestigationCorpus()