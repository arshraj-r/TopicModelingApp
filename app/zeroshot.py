import pandas as pd
from transformers import pipeline
import torch


class ZeroShot:
    def __init__(self, model_name="facebook/bart-large-mnli", device="cpu"):
        """Initialize the zero-shot classification pipeline."""
        self.device = 0 if device == "cuda" else -1  # 0 for GPU, -1 for CPU
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)

    def classify_texts(self, df, text_column, labels):
        """
        Classify text in a dataframe column using zero-shot classification.

        Args:
            df (pd.DataFrame): Input dataframe.
            text_column (str): Column name containing text to classify.
            labels (list): List of categories for classification.

        Returns:
            pd.DataFrame: Dataframe with probabilities for each label.
        """
        results = self.classifier(df[text_column].tolist(), candidate_labels=labels, multi_label=True)

        # Create new columns for each label and store the correct probability values
        for label in labels:
            df[label] = [next(score for l, score in zip(r["labels"], r["scores"]) if l == label) for r in results]

        return df
