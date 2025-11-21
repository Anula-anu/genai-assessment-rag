import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = "data/shl_catalog_clean.csv"
EMB_PATH = "data/shl_catalog_embeddings.npz"

def build_embeddings():
    df = pd.read_csv(CATALOG_PATH)
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    texts = df["use_cases"].astype(str).tolist()
    vectors = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.savez(EMB_PATH, embeddings=vectors)
    print("Saved embeddings to:", EMB_PATH)

if __name__ == "__main__":
    build_embeddings()
