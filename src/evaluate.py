from pathlib import Path

import pandas as pd

from retriever import CatalogRetriever

DATASET_PATH = Path("data/Gen_AI Dataset.xlsx")   # original file
TOP_K_LIST = [1, 3, 5]                            # metrics: Hit@1, Hit@3, Hit@5


def evaluate(top_k_list=TOP_K_LIST, max_rows=None):
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_excel(DATASET_PATH)

    # Expect columns: "Query" and "Assessment_url"
    df = df[["Query", "Assessment_url"]].dropna()

    if max_rows is not None:
        df = df.head(max_rows)

    retriever = CatalogRetriever()

    total = len(df)
    hits = {k: 0 for k in top_k_list}
    ranks = []

    print(f"Evaluating on {total} queries ...")

    for idx, row in df.iterrows():
        query = str(row["Query"])
        gold_url = str(row["Assessment_url"]).strip()

        # Retrieve candidates
        result_df = retriever.retrieve(query, top_k=max(top_k_list))
        cand_urls = [str(u).strip() for u in result_df["Assessment_url"].tolist()]

        # Compute rank of gold_url (if present)
        rank = None
        for i, u in enumerate(cand_urls, start=1):
            if u == gold_url:
                rank = i
                break
        if rank is not None:
            ranks.append(rank)

        # Hit@k
        for k in top_k_list:
            if gold_url in cand_urls[:k]:
                hits[k] += 1

    print("\n=== Retrieval Evaluation Results ===")
    for k in top_k_list:
        hit_k = hits[k] / total if total > 0 else 0.0
        print(f"Hit@{k}: {hit_k:.3f}  ({hits[k]} / {total})")

    if ranks:
        mrr = sum(1.0 / r for r in ranks) / len(ranks)
        print(f"MRR (Mean Reciprocal Rank): {mrr:.3f}  (on {len(ranks)} queries where gold was retrieved)")
    else:
        print("MRR: gold assessment was never retrieved in the top-k candidates.")


if __name__ == "__main__":
    # You can change max_rows to limit evaluation size, e.g., 200
    evaluate(max_rows=200)
