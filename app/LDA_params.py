import streamlit as st

def get_lda_params():
    st.title("LDA Hyperparameter Selection (Scikit-learn Defaults)")

    # Number of Topics
    num_topics = st.slider("Number of Topics (n_components)", min_value=2, max_value=50, value=10, step=1)

    # Alpha (Document-Topic Prior) - None means it's learned by the model
    alpha = st.slider("Alpha (Dirichlet Prior for Document-Topic Distribution, None = learned)", 
                      min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Beta (Topic-Word Prior) - None means it's learned by the model
    beta = st.slider("Beta (Dirichlet Prior for Topic-Word Distribution, None = learned)", 
                     min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Number of Iterations
    max_iter = st.slider("Number of Iterations (max_iter)", min_value=10, max_value=500, value=10, step=10)

    # Learning Method
    learning_method = st.radio("Learning Method", options=["batch", "online"], index=0)  # Default is 'batch'

    # Learning Decay (for online learning)
    learning_decay = st.slider("Learning Decay", min_value=0.5, max_value=1.0, value=0.7, step=0.05)

    # Display selected values
    st.subheader("Selected Hyperparameters:")
    st.write(f"**Number of Topics:** {num_topics}")
    st.write(f"**Alpha:** {alpha if alpha != 'None' else 'Learned by Model'}")
    st.write(f"**Beta:** {beta if beta != 'None' else 'Learned by Model'}")
    st.write(f"**Number of Iterations:** {max_iter}")
    st.write(f"**Learning Method:** {learning_method}")
    st.write(f"**Learning Decay:** {learning_decay}")

    # Return selected values
    return {
        "n_components": num_topics,
        "doc_topic_prior": None if alpha == 'None' else alpha,
        "topic_word_prior": None if beta == 'None' else beta,
        "max_iter": max_iter,
        "learning_method": learning_method,
        "learning_decay": learning_decay
    }
