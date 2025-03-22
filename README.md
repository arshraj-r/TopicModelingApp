# TopicModelingToolkit

**TopicModelingApp** is an interactive web application designed for topic modeling using a variety of algorithms, including **BERTopic**, **LDA**, and **scikit-learn**. It allows users to upload text data, run topic modeling, and visualize the extracted topics in an easy-to-use interface built with **Streamlit**. The app also supports cloud deployment via **Docker** and is intended to be easily deployed on cloud platforms such as **Google Cloud**, **AWS**, or **Kubeflow**.

---

## Features

- **Topic Modeling**: Use popular models such as **BERTopic**, **LDA**, and **scikit-learn's NMF** to extract topics from text data.
- **Interactive UI**: A **Streamlit** app that allows users to upload data, run topic modeling, and view results in real time.
- **Visualizations**: Generate insightful visualizations like word clouds, bar charts, and topic distributions to help interpret the topics.
- **FastAPI Backend**: An optional backend API to serve the topic modeling results for integration with other applications.
- **Dockerized Application**: The app is fully containerized using **Docker**, making it easy to deploy anywhere.
- **Cloud-Ready**: Easily deployable on **Google Cloud**, **AWS**, or any other cloud platform using Docker or Kubernetes.

---

## Installation

### Prerequisites

Before running the app locally, you need to have the following installed:

- **Python 3.8+**
- **Docker** (for containerization and deployment)
- **Streamlit** (for the app's front-end)
- **FastAPI** (for optional backend API)

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TopicModelingApp.git
cd TopicModelingApp
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/main.py
```

This will open the Streamlit app in your web browser, where you can upload your text data, select a model, and view the topics and visualizations.

---

## Usage

### Topic Modeling Algorithms

- **BERTopic**: A transformer-based topic modeling approach that leverages BERT embeddings, UMAP for dimensionality reduction, and HDBSCAN for clustering.
- **LDA (Latent Dirichlet Allocation)**: A classical topic modeling approach that models topics as distributions over words and documents as mixtures of topics.
- **Scikit-learn's NMF**: Non-Negative Matrix Factorization is another method for topic modeling, focusing on factorizing the document-term matrix.

### How to Use the App

1. Upload your text data (e.g., CSV, TXT files).
2. Select a topic modeling algorithm: **BERTopic**, **LDA**, or **NMF**.
3. Run the model to extract topics.
4. View the generated topics and their top words.
5. Visualize the results using word clouds, bar charts, or topic distributions.

---

## Docker Deployment

To containerize the app and deploy it on any cloud platform, you can use Docker.

### Build the Docker Image

```bash
docker build -t topic-modeling-app .
```

### Run the Docker Container Locally

```bash
docker run -p 8501:8501 topic-modeling-app
```

This will run the app locally on port 8501.

### Deploy to Google Cloud / Kubernetes / AWS

Once the app is containerized with Docker, you can easily deploy it using **Google Cloud Run**, **AWS ECS**, or any Kubernetes-based service. For specific deployment instructions, check out the documentation for the respective cloud platform.

---

## FastAPI Integration (Optional)

If you'd like to use the **FastAPI** backend to serve the topic modeling results, follow these steps:

1. Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

2. Access the endpoints to get topic modeling results:
   - POST `/lda` to run the LDA model.
   - POST `/bertopic` to run the BERTopic model.

The FastAPI backend can be used in conjunction with Streamlit or other applications to integrate topic modeling results programmatically.

---

## Folder Structure

```
TopicModelingApp/
├── app/
│   ├── __init__.py             # Initialize the app module
│   ├── main.py                 # Main Streamlit app or FastAPI app
│   ├── topic_modeling.py       # Core logic for topic modeling (BERTopic, LDA, NMF)
│   ├── utils.py                # Utility functions for data preprocessing
│   ├── visualizations.py       # Code for generating visualizations
│   └── config.py               # Configuration file for model parameters
├── models/                     # Store pre-trained models here
│   ├── bertopic_model/         # Saved BERTopic model
│   ├── lda_model/              # Saved LDA model
│   └── sklearn_model/          # Saved sklearn models (e.g., NMF)
├── data/                       # Data input/output directory
│   ├── raw/                    # Raw data before processing
│   ├── processed/              # Processed data (after cleaning)
│   └── output/                 # Output files such as topic reports, results
├── requirements.txt            # List of dependencies
├── Dockerfile                  # Docker configuration for containerization
├── README.md                   # Project overview and setup instructions
├── setup.py                    # Optional for packaging
└── tests/                      # Unit tests for models, app, utils
    ├── test_topic_modeling.py  # Tests for topic modeling functionality
    ├── test_visualizations.py  # Tests for visualizations
    ├── test_app.py             # Tests for app functionality
    └── test_utils.py           # Tests for utility functions
```

---

## Contributing

We welcome contributions to this project! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Create a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you'd like any additional sections or further modifications!