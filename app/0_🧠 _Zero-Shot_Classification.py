import streamlit as st
import pandas as pd
from zeroshot import ZeroShot
import torch
torch.cuda.empty_cache()

st.set_page_config(
    page_title="Zero-Shot Classification",
    page_icon="🧠",
    layout="wide"
)


# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Display device info in the Streamlit app
if device == "cuda":
    st.success("🚀 Running on **GPU (CUDA)** for faster inference!")
else:
    st.warning("⚠️ Running on **CPU**. Inference might be slower.")



st.title("🧠  Zero-Shot Classification App")
st.write("Upload a CSV file and classify text into custom categories using Hugging Face.")

# Model selection
model_options = {
    "facebook/bart-large-mnli": "BART-Large (Best for general use)",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli": "mDeBERTa (Multilingual Support)",
    "valhalla/distilbart-mnli-12-3": "DistilBART (Faster & lightweight)",
}

model_name = st.selectbox("Select Zero-Shot Model:", list(model_options.keys()), format_func=lambda x: model_options[x])

# Initialize classifier
classifier = ZeroShot(model_name=model_name,device=device)

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Select text column
    text_column = st.selectbox("Select Text Column:", df.columns)

    # Enter categories
    labels = st.text_input("Enter Categories (comma-separated):", "Business, Technology, Sports, Health")
    labels = [label.strip() for label in labels.split(",")]

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            df = classifier.classify_texts(df, text_column, labels)

        st.write("### Classification Results", df.head())

        # Allow CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "classified_results.csv", "text/csv")

