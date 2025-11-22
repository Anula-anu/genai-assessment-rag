from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path("data/shl_catalog_clean.csv")
SCRAPED_PATH = Path("data/shl_web_catalog.csv")
EMBEDDINGS_PATH = Path("data/shl_catalog_embeddings.npz")


def build_embeddings():
    print(f"Loading base catalog from {CATALOG_PATH} ...")
    df = pd.read_csv(CATALOG_PATH)

    texts_col = "use_cases"

    if SCRAPED_PATH.exists():
        print(f"Found scraped web catalog at {SCRAPED_PATH}, merging...")
        web_df = pd.read_csv(SCRAPED_PATH)

        df = df.merge(
            web_df,
            left_on="Assessment_url",
            right_on="product_url",
            how="left",
        )

        df["combined_text"] = (
            df["use_cases"].fillna("") + " "
            + df["product_title"].fillna("") + " "
            + df["meta_description"].fillna("") + " "
            + df["page_text"].fillna("")
        )

        texts_col = "combined_text"

    texts = df[texts_col].fillna("").tolist()

    print(f"Encoding {len(texts)} items using SentenceTransformer...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        assessment_urls=df["Assessment_url"].values,
    )
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_embeddings()
