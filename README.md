# GenAI Assessment Recommendation Engine (RAG)

This project was built as part of the **SHL Research Intern Assessment**.  
It implements a **Retrieval-Augmented Generation (RAG)** system using SHL’s product catalog to recommend the most relevant assessments based on a job description or hiring need.

---

## 🚀 Project Overview

The system performs the following steps:

1. **Reads SHL Product Catalog Dataset**  
   Cleans, preprocesses, and structures the dataset into a catalog.

2. **Embeds Use-Cases with SentenceTransformer**  
   Uses the `all-MiniLM-L6-v2` model to generate vector embeddings.

3. **Builds a Vector Index for Similarity Search**  
   Retrieves the top-k similar historical use-cases to the user’s job description.

4. **Maps Retrieved Results to Assessments**  
   Shows the recommended assessments with similarity scores.

5. **Optional LLM Explanation (Fallback Enabled)**  
   If OpenAI API quota is available → LLM explains the recommendations.  
   If quota is exhausted → Fallback text-based explanation is returned.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | Python |
| UI | Streamlit |
| Embeddings | SentenceTransformer |
| Retrieval | Cosine Similarity |
| RAG Explanation | OpenAI GPT-4o (optional) |
| Storage | CSV (catalog + embeddings) |
| Environment | Virtualenv |

---

## 📂 Project Structure

genai-assessment-rag/
│
├── data/
│ ├── Gen_AI Dataset.xlsx # Original dataset
│ ├── shl_catalog_clean.csv # Cleaned catalog generated
│ ├── embeddings.npy # Embeddings file (generated)
│
├── src/
│ ├── make_catalog.py # Cleans and builds catalog
│ ├── indexer.py # Generates embeddings
│ ├── retriever.py # Retrieves similar assessments
│ ├── rag_engine.py # RAG engine + explanation module
│ ├── app_streamlit.py # Streamlit web app
│
├── .gitignore
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anula-anu/genai-assessment-rag.git
cd genai-assessment-rag
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/Scripts/activate   # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
📦 Build the Catalog & Embeddings
Build Catalog
bash
Copy code
python src/make_catalog.py
Build Embeddings
bash
Copy code
python src/indexer.py
▶️ Run the Streamlit App
bash
Copy code
streamlit run src/app_streamlit.py
The app will open at:

arduino
Copy code
http://localhost:8501
🔑 OpenAI API Key (Optional)
The explanation feature uses GPT-4o.
Set your API key only if quota is available:

bash
Copy code
set OPENAI_API_KEY=your_key_here        # Windows
If no key is set or quota is exhausted, the app automatically falls back to a rule-based explanation.

📊 Features Demonstrated (For SHL Review)
✔ Full RAG pipeline implementation
✔ Vector search using embeddings
✔ Clean UI with Streamlit
✔ Error handling + fallback logic
✔ Modular, production-style code structure
✔ Works even without LLM quota

📝 Author
Anula Biju
GitHub: https://github.com/Anula-anu
