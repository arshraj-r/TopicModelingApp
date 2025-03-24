import pandas as pd
import streamlit as st

def get_bertopic_params():
    st.subheader("BERTopic Hyperparameter Selection")

    # General Parameters
    n_gram_range = st.radio("N-gram Range", options=[(1,1), (1,2), (1,3)], index=1,horizontal=True)
    top_n_words = st.slider("Top N Words", min_value=5, max_value=50, value=10, step=1)
    min_topic_size = st.slider("Minimum Topic Size", min_value=5, max_value=100, value=10, step=5)
    nr_topics = st.slider("Number of Topics (None = auto)", min_value=2, max_value=50, value=10, step=1)
    calculate_probabilities = st.checkbox("Calculate Probabilities", value=False)
    diversity = st.slider("Diversity (0 = similar, 1 = diverse)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # UMAP Parameters
    n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=50, value=15, step=1)
    n_components = st.slider("Number of Components", min_value=2, max_value=10, value=5, step=1)
    metric = st.radio("Distance Metric", options=["cosine", "euclidean", "manhattan"], index=0,horizontal=True)

    # HDBSCAN Parameters
    min_cluster_size = st.slider("Minimum Cluster Size", min_value=2, max_value=50, value=10, step=1)
    cluster_selection_method = st.radio("Cluster Selection Method", options=["eom", "leaf"], index=0,horizontal=True)

    return {
        "n_gram_range": n_gram_range,
        "top_n_words": top_n_words,
        "min_topic_size": min_topic_size,
        "nr_topics": None if nr_topics == 10 else nr_topics,
        "calculate_probabilities": calculate_probabilities,
        "diversity": diversity,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "metric": metric,
        "min_cluster_size": min_cluster_size,
        "cluster_selection_method": cluster_selection_method
    }

def get_lda_params():
    st.subheader("LDA Hyperparameter Selection (Scikit-learn Defaults)")

    # Number of Topics
    num_topics = st.slider("Number of Topics (n_components)", min_value=2, max_value=50, value=10, step=1)
    st.divider()
    # Alpha (Document-Topic Prior) - None means it's learned by the model
    # Alpha (Document-Topic Prior)
    alpha_option = st.radio("Alpha (Dirichlet Prior for Document-Topic Distribution)", options=["None", "Custom"], index=0,horizontal=True)
    alpha = None if alpha_option == "None" else st.slider("Alpha Value", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    st.divider()

    # Beta (Topic-Word Prior)
    beta_option = st.radio("Beta (Dirichlet Prior for Topic-Word Distribution)", options=["None", "Custom"], index=0,horizontal=True)
    beta = None if beta_option == "None" else st.slider("Beta Value", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    st.divider()

    # Number of Iterations
    max_iter = st.slider("Number of Iterations (max_iter)", min_value=10, max_value=500, value=10, step=10)

    st.divider()
    # Learning Method
    learning_method = st.radio("Learning Method", options=["batch", "online"], index=0,horizontal=True)  # Default is 'batch'

    st.divider()

    # Learning Decay (for online learning)
    learning_decay = st.slider("Learning Decay", min_value=0.5, max_value=1.0, value=0.7, step=0.05)

    # Display selected values
    # st.subheader("Selected Hyperparameters:")
    # st.write(f"**Number of Topics:** {num_topics}")
    # st.write(f"**Alpha:** {alpha if alpha != 'None' else 'Learned by Model'}")
    # st.write(f"**Beta:** {beta if beta != 'None' else 'Learned by Model'}")
    # st.write(f"**Number of Iterations:** {max_iter}")
    # st.write(f"**Learning Method:** {learning_method}")
    # st.write(f"**Learning Decay:** {learning_decay}")

    # Return selected values
    return {
        "n_components": num_topics,
        "doc_topic_prior": None if alpha == 'None' else alpha,
        "topic_word_prior": None if beta == 'None' else beta,
        "max_iter": max_iter,
        "learning_method": learning_method,
        "learning_decay": learning_decay
    }


# Function to save the result as an Excel file
def save_to_excel(df, filename="output.xlsx"):
    """
    Save the DataFrame to an Excel file.

    Parameters:
    df (pandas DataFrame): DataFrame containing the results to be saved.
    filename (str): The output Excel file name.

    Returns:
    str: The filename of the saved Excel file.
    """
    df.to_excel(filename, index=False)
    return filename