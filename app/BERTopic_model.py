from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"

# Create your representation model


class TopicModeling:
    def __init__(self):
        """
        Initializes the TopicModeling class with the necessary models for BERTopic.
        """
        # Step 1 - Extract embeddings using SentenceTransformer
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Step 2 - Reduce dimensionality with UMAP
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

        # Step 3 - Cluster reduced embeddings with HDBSCAN
        self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # Step 4 - Tokenize topics using CountVectorizer
        self.vectorizer_model = CountVectorizer(stop_words="english")

        # Step 5 - Topic representations using ClassTfidfTransformer
        self.ctfidf_model = ClassTfidfTransformer()

        # Step 6 - (Optional) Fine-tune topic representations using KeyBERTInspired
        # self.representation_model = KeyBERTInspired()
        generator = pipeline('text2text-generation', model='google/flan-t5-base')
        # generator = pipeline('text2text-generation', model="microsoft/Phi-3.5-mini-instruct")
        self.representation_model = TextGeneration(generator)

        # Initialize BERTopic model
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,          # Step 1
            umap_model=self.umap_model,                    # Step 2
            hdbscan_model=self.hdbscan_model,              # Step 3
            vectorizer_model=self.vectorizer_model,        # Step 4
            ctfidf_model=self.ctfidf_model,                # Step 5
            representation_model=self.representation_model # Step 6
        )

    def fit(self, documents):
        """
        Fits the BERTopic model on a list of documents.
        
        Parameters:
        - documents: A list of text documents to model.
        
        Returns:
        - topics: The topics assigned to each document.
        - probabilities: The probabilities associated with each topic.
        """
        topics, probabilities = self.topic_model.fit_transform(documents)
        return topics, probabilities

    def get_topics(self):
        """
        Returns the topics identified by the model along with their top words.
        
        Returns:
        - topics: A list of topics and their associated top words.
        """
        return self.topic_model.get_topic_info()

    def get_topic_words(self, topic_id):
        """
        Returns the words associated with a specific topic.
        
        Parameters:
        - topic_id: The ID of the topic.
        
        Returns:
        - words: The top words for the given topic.
        """
        return self.topic_model.get_topic(topic_id)

    def visualize_topics(self):
        """
        Generates a visualization of the topics.
        
        Returns:
        - visualization: A visualization object (interactive plot) showing the topics.
        """
        return self.topic_model.visualize_topics()

    def visualize_barchart(self):
        """
        Generates a bar chart of the topics.
        
        Returns:
        - visualization: A bar chart object of the topics.
        """
        return self.topic_model.visualize_barchart()

    def visualize_hierarchy(self):
        """
        Generates a hierarchical tree of the topics.
        
        Returns:
        - visualization: A hierarchical tree showing topic relationships.
        """
        return self.topic_model.visualize_hierarchy()

    def save_model(self, path):
        """
        Saves the fitted BERTopic model to disk.
        
        Parameters:
        - path: The directory path where the model should be saved.
        """
        self.topic_model.save(path)

    def load_model(self, path):
        """
        Loads a saved BERTopic model from disk.
        
        Parameters:
        - path: The directory path from which the model should be loaded.
        """
        self.topic_model = BERTopic.load(path)
