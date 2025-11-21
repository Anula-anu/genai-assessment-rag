import pandas as pd

INPUT_PATH = "data/Gen_AI Dataset.xlsx"
OUTPUT_PATH = "data/shl_catalog_clean.csv"

def build_catalog():
    df = pd.read_excel(INPUT_PATH)

    df["Query"] = df["Query"].astype(str).str.strip()
    df["Assessment_url"] = df["Assessment_url"].astype(str).str.strip()

    grouped = (
        df.groupby("Assessment_url")["Query"]
        .apply(lambda x: " || ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"Query": "use_cases"})
    )

    grouped.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved catalog to: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_catalog()
