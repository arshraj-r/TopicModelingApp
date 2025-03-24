import streamlit as st

def get_bertopic_params():
    st.title("BERTopic Hyperparameter Selection")

    # General Parameters
    n_gram_range = st.radio("N-gram Range", options=[(1,1), (1,2), (1,3)], index=1)
    top_n_words = st.slider("Top N Words", min_value=5, max_value=50, value=10, step=1)
    min_topic_size = st.slider("Minimum Topic Size", min_value=5, max_value=100, value=10, step=5)
    nr_topics = st.slider("Number of Topics (None = auto)", min_value=2, max_value=50, value=10, step=1)
    calculate_probabilities = st.checkbox("Calculate Probabilities", value=False)
    diversity = st.slider("Diversity (0 = similar, 1 = diverse)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # UMAP Parameters
    n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=50, value=15, step=1)
    n_components = st.slider("Number of Components", min_value=2, max_value=10, value=5, step=1)
    metric = st.radio("Distance Metric", options=["cosine", "euclidean", "manhattan"], index=0)

    # HDBSCAN Parameters
    min_cluster_size = st.slider("Minimum Cluster Size", min_value=2, max_value=50, value=10, step=1)
    cluster_selection_method = st.radio("Cluster Selection Method", options=["eom", "leaf"], index=0)

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
