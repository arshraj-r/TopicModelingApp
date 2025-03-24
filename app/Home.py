import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Topic Modeling Project",
    page_icon="📖",
    layout="wide"
)

# Title
st.title("🔍 Topic Modeling Exploration")

# Description
st.markdown("""
### About This Project
This project focuses on **Topic Modeling**, an unsupervised learning technique used to uncover hidden themes in textual data.

#### **Current Implementations:**
- **LDA (Latent Dirichlet Allocation)** 📖  
- **BERTopic Modeling** 🤖  

#### **Future Work:**
- **Guided Topic Modeling** 🏷️ (Semi-supervised techniques for better control)  
- **Semi-supervised Topic Modeling** 🔬 (Combining labeled & unlabeled data)  

This application allows users to explore different topic modeling approaches and visualize the extracted topics. 🚀
""")

# Footer
st.info("Navigate through the sidebar to explore different models!")

