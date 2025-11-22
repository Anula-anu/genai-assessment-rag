# 🧠 GenAI Assessment Recommendation Engine (RAG)

This project was built as part of the **SHL Research Intern Assessment**.  
It implements a **Retrieval-Augmented Generation (RAG)** engine using SHL’s product catalog to recommend the most relevant assessments for a given **job description or hiring need**.

---

## 🚀 Project Objective
To assist hiring teams by intelligently mapping job descriptions to the most suitable SHL assessments using:
- Web-scraped SHL product metadata
- Semantic similarity using embeddings
- RAG + LLM-based explanation for recommendations

---

## 🏗️ System Pipeline Overview

| Stage | Description |
|-------|-------------|
| **1. Catalog Construction** | Cleans and preprocesses SHL product catalog into structured format (`make_catalog.py`) |
| **2. Web Scraping** | Scrapes product pages (title, metadata, descriptions) from SHL’s website to enrich dataset (`scraper.py`) |
| **3. Semantic Indexing** | Generates SentenceTransformer embeddings of use-cases and stores them (`indexer.py`) |
| **4. Retrieval Engine** | Computes similarity between user query and catalog use-cases (`retriever.py`) |
| **5. Recommendation + LLM Explanation** | Returns top matching assessments and an AI-generated explanation via OpenAI API (`rag_engine.py` + `app_streamlit.py`) |

---

## 🖥️ Web Application (Streamlit)
The frontend allows recruiters to enter a hiring requirement and receive:
- Recommended SHL assessments
- Example historical use cases
- Similarity scores
- AI-generated summarized explanation

To run locally:

```bash
pip install -r requirements.txt
streamlit run src/app_streamlit.py
📊 Evaluation Results
The RAG system was evaluated using SHL’s historical dataset (65 queries):

Metric	Score
Hit@1	0.154
Hit@3	0.462
Hit@5	0.646
MRR (Mean Reciprocal Rank)	0.502 (on 42 queries where gold label was retrieved)

These results show that the system retrieves highly relevant assessments for most hiring descriptions.

Run evaluation manually:

bash
Copy code
python src/evaluate.py
📂 Project Structure
css
Copy code
├── data/
│   ├── shl_catalog_clean.csv
│   ├── shl_web_catalog.csv                ← scraped metadata
│   └── shl_catalog_embeddings.npz         ← semantic index
│
├── src/
│   ├── make_catalog.py
│   ├── scraper.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── rag_engine.py
│   └── app_streamlit.py
│
├── requirements.txt
└── README.md
🔧 Tech Stack
Component	Tool
Web Scraping	BeautifulSoup, Requests
Embeddings	SentenceTransformer (MiniLM)
Retrieval	Cosine similarity
LLM	OpenAI API
Frontend	Streamlit
Evaluation	Hit@k, MRR

🌱 Future Enhancements
Support multilingual job descriptions

Fine-tuning model with SHL domain data

Add hybrid search (BM25 + embeddings)

📝 Notes
API key must be stored in the environment variable:

arduino
Copy code
export OPENAI_API_KEY="your_key_here"
Do not commit API keys to GitHub.

🙌 Acknowledgment
This project was independently implemented for the SHL Research Intern Assessment with the goal of demonstrating RAG techniques for HR technology.

🔗 Github Repository
https://github.com/Anula-anu/genai-assessment-rag

---

After pasting the README:
1. Commit changes
2. Push to GitHub
3. Submit the GitHub URL in the SHL portal
4. Select **YES** for all requirements ✔️

If you want, I can also generate **a short video demo script** to record your project explanation for extra impression 🌟


yaml
Copy code
