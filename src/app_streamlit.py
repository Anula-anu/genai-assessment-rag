import streamlit as st
from rag_engine import GenAIRAGEngine

st.set_page_config(page_title="GenAI Assessment Recommendation", layout="wide")

st.title("GenAI Assessment Recommendation")

st.markdown(
    "Enter a job description or hiring need. "
    "The system will retrieve similar historical queries from the dataset, "
    "map them to assessments, and use GenAI to explain the recommendations."
)

job_desc = st.text_area("Enter Job Description", height=150)

engine = GenAIRAGEngine()

if st.button("Recommend"):
    if not job_desc.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Retrieving and generating explanation..."):

            explanation, retrieved = engine.recommend_with_explanation(
                job_description=job_desc,
                top_k_retrieve=8,
            )

        st.subheader("GenAI Explanation")
        st.markdown(explanation)

        st.subheader("Retrieved Assessments from Dataset")
        st.dataframe(retrieved[["Assessment_url", "use_cases", "similarity"]])
