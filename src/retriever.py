import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class CatalogRetriever:
    def __init__(self):
        self.df = pd.read_csv("data/shl_catalog_clean.csv")
        self.embeddings = np.load("data/shl_catalog_embeddings.npz")["embeddings"]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, top_k=5):
        q_vec = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_vec, self.embeddings)[0]

        df = self.df.copy()
        df["similarity"] = sims

        return df.sort_values("similarity", ascending=False).head(top_k)
