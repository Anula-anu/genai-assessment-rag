from retriever import CatalogRetriever

def build_prompt(job_desc, df_rows):
    text = ""
    for _, row in df_rows.iterrows():
        text += f"\nAssessment: {row['Assessment_url']}\nUse Cases: {row['use_cases']}\n"

    prompt = f"""
You are an expert in assessments.

Job description:
{job_desc}

Based on past use-cases:
{text}

Recommend the best assessments and explain why.
"""

    return prompt

def call_llm(prompt):
    raise NotImplementedError("Add an LLM call here later")

class GenAIRAGEngine:
    def __init__(self):
        self.retriever = CatalogRetriever()

    def recommend(self, job_desc):
        df = self.retriever.retrieve(job_desc, top_k=5)
        prompt = build_prompt(job_desc, df)
        answer = call_llm(prompt)
        return answer, df
