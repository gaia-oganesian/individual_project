import os
import pandas as pd
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Tuple, Optional, List

def setup_environment(env_file: str = '.env') -> None:
    """
    Load environment variables from a .env file and set necessary environment variables.
    
    Args:
        env_file (str): Path to the .env file. Defaults to '.env'.
    """
    load_dotenv(env_file)
    os.environ['CUDA_LAUNCH_BLOCKING'] = os.getenv('CUDA_LAUNCH_BLOCKING', '1')
    login(token=os.getenv('HF_TOKEN'))

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    """
    df.to_csv(file_path, index=False)

def initialize_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load the tokenizer and model from Hugging Face.
    
    Args:
        model_name (str): The Hugging Face model name (e.g., 'ProsusAI/finbert' or 'ElKulako/cryptobert').
    
    Returns:
        AutoTokenizer, AutoModelForSequenceClassification: The tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model

def get_sentiment(text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> int:
    """
    Predict the sentiment of a given text using a pre-trained model.
    
    Args:
        text (str): The input text to analyze.
        tokenizer: The tokenizer for the pre-trained model.
        model: The pre-trained model for sequence classification.
    
    Returns:
        int: The predicted sentiment label (0 = negative, 1 = neutral, 2 = positive).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(probs, dim=1).item()

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_column: str = 'sentiment_label') -> None:
    """
    Plot the distribution of sentiment labels.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sentiment data.
        sentiment_column (str): The name of the column containing sentiment labels.
    """
    sns.countplot(x=sentiment_column, data=df)
    plt.title('Sentiment Distribution')
    plt.show()

def plot_category_sentiment_distribution(df: pd.DataFrame, category_column: str = 'category_label', sentiment_column: str = 'sentiment_label', desired_order: Optional[List[str]] = None) -> None:
    """
    Plot the distribution of sentiment labels per category.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sentiment and category data.
        category_column (str): The name of the column containing category labels.
        sentiment_column (str): The name of the column containing sentiment labels.
        desired_order (list): The desired order for categories.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(
        x=category_column,
        hue=sentiment_column,
        data=df,
        order=desired_order
    )
    plt.title('Sentiment Distribution per Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')

    # Add percentages on top of each bar
    total_counts = len(df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # To avoid displaying 0%
            percentage = f'{height / total_counts * 100:.1f}%'
            ax.annotate(
                percentage,
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
    
    plt.show()
