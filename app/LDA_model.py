import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDATopicModel:
    def __init__(self, num_topics=5, max_features=1000):
        self.num_topics = num_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        self.model = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)

    def fit(self, text_data, return_doc_topics=False):
        text_data = text_data.dropna()  # Remove missing values
        doc_term_matrix = self.vectorizer.fit_transform(text_data)
        self.model.fit(doc_term_matrix)

        topics = self.get_topics()

        if return_doc_topics:
            doc_topics = self.model.transform(doc_term_matrix)
            predicted_topics = doc_topics.argmax(axis=1)  # Most probable topic for each document
            topic_probabilities = doc_topics.max(axis=1)  # Probability of the most probable topic
            return topics, predicted_topics, topic_probabilities

        return topics

    def get_topics(self, n_words=10):
        words = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            topics[f"Topic {topic_idx+1}"] = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        return topics
