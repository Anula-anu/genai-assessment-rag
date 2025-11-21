import textwrap

from retriever import CatalogRetriever

# If later you add billing, you can uncomment these lines and use real OpenAI
# from openai import OpenAI, OpenAIError
# client = OpenAI()


def build_prompt(job_desc, df_rows):
    chunks = []
    for _, row in df_rows.iterrows():
        chunk = f"""
        Assessment URL: {row['Assessment_url']}
        Historical use-cases:
        {row['use_cases']}
        Similarity score: {row['similarity']:.3f}
        """
        chunks.append(textwrap.dedent(chunk).strip())

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""
You are an expert in SHL assessments.

Job Description:
\"\"\"{job_desc}\"\"\"

Based on the historical use-cases:
{context}

Task:
1. Select the 3–5 most suitable assessments from the above list.
2. Explain why each one fits the job description.
3. Mention what skills/competencies it measures.
4. Be concise, clear, and structured.

Do NOT invent new assessments.
"""
    return textwrap.dedent(prompt).strip()


def call_llm(prompt: str) -> str:
    """
    Real LLM call – currently disabled because of API quota.
    If you later add billing and want to use GPT-4o-mini,
    uncomment the OpenAI imports above and this function body.
    """
    # Example (when you have credits):
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are an experienced SHL assessment consultant."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.3
    # )
    # return response.choices[0].message.content
    raise RuntimeError("LLM disabled (no API quota).")


def build_fallback_explanation(job_description, df_top) -> str:
    """
    Rule-based 'GenAI-like' explanation used when LLM is unavailable.
    Uses the retrieved assessments and their historical use-cases.
    """
    lines = []
    lines.append("**Note:** LLM API quota is exhausted, so this explanation is generated ")
    lines.append("using a simple rule-based summarization over the retrieved assessments.\n")

    lines.append("### Summary\n")
    lines.append(
        f"For the hiring need:\n\n> {job_description}\n\n"
        "the following assessments have been frequently used in similar historical queries "
        "from the dataset and are therefore likely to be relevant.\n"
    )

    lines.append("### Recommended assessments (from retrieved results)\n")

    for i, (_, row) in enumerate(df_top.head(5).iterrows(), start=1):
        # Take the first use-case as a short description
        first_usecase = str(row["use_cases"]).split("||")[0].strip()
        first_usecase = first_usecase[:220] + ("..." if len(first_usecase) > 220 else "")

        lines.append(f"{i}. **Assessment URL:** {row['Assessment_url']}")
        lines.append(f"   - Example historical use-case: _{first_usecase}_")
        lines.append(
            f"   - Similarity score (to your description): `{row['similarity']:.3f}`\n"
        )

    return "\n".join(lines)


class GenAIRAGEngine:
    def __init__(self):
        self.retriever = CatalogRetriever()

    def recommend_with_explanation(self, job_description: str, top_k_retrieve: int = 8):
        # 1. Retrieve top-k assessments
        df_top = self.retriever.retrieve(job_description, top_k=top_k_retrieve)

        # 2. Try LLM (currently disabled) and fall back gracefully
        prompt = build_prompt(job_description, df_top)
        try:
            explanation = call_llm(prompt)
        except Exception:
            # When LLM fails (no quota, no key, etc.), use fallback explanation
            explanation = build_fallback_explanation(job_description, df_top)

        return explanation, df_top
