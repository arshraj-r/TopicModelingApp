import streamlit as st
import pandas as pd
import sys
import os
import xlsxwriter
from LDA_params import get_lda_params

from LDA_model import LDATopicModel

st.set_page_config(
    page_title="LDA Topic Modeling",
    page_icon="📖",
    layout="wide"
)

st.title("📖 LDA Topic Modeling")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
# num_topics = st.number_input("Enter number of topics", min_value=2, max_value=20, value=5)



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select a column for topic modeling", columns)
    params = get_lda_params()
    start_button = st.button("Start Topic Modeling")

    if start_button and selected_column:
        lda_model = LDATopicModel(num_topics=params["n_components"])
        topics, doc_topics, topic_probabilities = lda_model.fit(df[selected_column], return_doc_topics=True)

        # Display identified topics
        st.write("### Identified Topics")
        for topic, words in topics.items():
            st.write(f"**{topic}:** {', '.join(words)}")

        # Create DataFrame for document-topic predictions
        df["Predicted Topic"] = doc_topics
        df["Topic Probability"] = topic_probabilities

        # Save results to Excel
        output_file = "LDA_Results.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Document Topics", index=False)
            pd.DataFrame(topics).T.to_excel(writer, sheet_name="Topic Details")

        st.success("Topic Modeling completed! Download the results below.")
        st.download_button(
            label="📥 Download Excel File",
            data=open(output_file, "rb"),
            file_name=output_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
