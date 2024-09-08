import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_model_and_tokenizer, predict_category, visualize_probabilities, category_mapping
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict

# Add the parent directory (project) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessing import process_text
import torch.nn.functional as F




# Load models and tokenizers
category_model_path = '../trained_models/bertft_optimizer_Adam_lr_1e-05_epochs_10_bs_8_maxlen_512'
category_model, category_tokenizer = load_model_and_tokenizer(category_model_path)

# Load CryptoBERT for sentiment analysis
sentiment_tokenizer = AutoTokenizer.from_pretrained('ElKulako/cryptobert')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('ElKulako/cryptobert')

# Define sentiment mapping
sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}



def visualize_probabilities(probabilities: torch.Tensor, category_mapping: Dict[int, str]) -> None:
    """
    Visualizes the predicted probabilities for event categories.

    Args:
        probabilities (torch.Tensor): The probabilities for each category.
        category_mapping (Dict[int, str]): A mapping of category indices to their names.
    """
    # Convert probabilities to a DataFrame for easier plotting
    categories = [category_mapping[i] for i in range(len(probabilities))]
    probabilities_df = pd.DataFrame({'Category': categories, 'Probability': probabilities})
    
    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(x='Probability', y='Category', hue='Category', data=probabilities_df, palette="viridis", dodge=False)
    
    # Add percentage labels to the bars
    for p in ax.patches:
        width = p.get_width()
        percentage = f'{width*100:.2f}%'
        x_offset = width + 0.01 if width < 0.03 else width - 0.02
        plt.text(x_offset, p.get_y() + p.get_height() / 2, percentage, ha='left' if width < 0.03 else 'center', va='center', color='black' if width < 0.03 else 'white')
    
    plt.xlabel('Probability')
    plt.ylabel('Category')
    plt.title('Predicted Probabilities by Category')

    # Use st.pyplot to render the plot in Streamlit
    st.pyplot(plt)


def visualize_sentiment_probabilities(probabilities: torch.Tensor, sentiment_mapping: Dict[int, str]) -> None:
    """
    Visualizes the predicted probabilities for sentiment analysis.

    Args:
        probabilities (torch.Tensor): The probabilities for each sentiment.
        sentiment_mapping (Dict[int, str]): A mapping of sentiment indices to their names.
    """
    # Convert probabilities to a DataFrame for easier plotting
    sentiments = [sentiment_mapping[i] for i in range(len(probabilities))]
    probabilities_df = pd.DataFrame({'Sentiment': sentiments, 'Probability': probabilities})
    
    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 3))
    ax = sns.barplot(x='Probability', y='Sentiment', data=probabilities_df, palette="viridis", dodge=False)
    
    # Add percentage labels to the bars
    for p in ax.patches:
        width = p.get_width()
        percentage = f'{width*100:.2f}%'
        x_offset = width + 0.01 if width < 0.03 else width - 0.02
        plt.text(x_offset, p.get_y() + p.get_height() / 2, percentage, ha='left' if width < 0.03 else 'center', va='center', color='black' if width < 0.03 else 'white')
    
    plt.xlabel('Probability')
    plt.ylabel('Sentiment')
    plt.title('Sentiment Probabilities')

    # Use st.pyplot to render the plot in Streamlit
    st.pyplot(plt)


# Define functions
def get_sentiment(text: str) -> Tuple[int, torch.Tensor]:
    """
    Predicts the sentiment for a given text.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        Tuple[int, torch.Tensor]: The predicted sentiment and its probabilities.
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=1).item()
    return sentiment, probs

# App title
st.title("Event Categorisation and Sentiment Analysis App")

# Input form
st.header("Enter Event Information")
event_title = st.text_input("Event Title (optional)")
event_text = st.text_area("Event Text (required)", height=200)

# Prediction button
if st.button("Predict"):
    if event_text.strip():
        # Preprocess the text
        processed_text = process_text(event_text)

        # Set the threshold for categorisation
        threshold = 0.7  

        # After predicting categories
        predicted_categories, probabilities_list = predict_category(category_model, category_tokenizer, [processed_text])
        probabilities = probabilities_list[0]

        # Check if the highest probability exceeds the threshold
        max_prob = max(probabilities[0])
        if max_prob < threshold:
            predicted_category_name = "Uncategorized"
        else:
            predicted_category = predicted_categories[0]
            predicted_category_name = category_mapping[predicted_category]


        # Sentiment Prediction
        sentiment, sentiment_probs = get_sentiment(processed_text)
        sentiment_label = sentiment_mapping[sentiment]

        # Display results
        st.subheader("Results")
        st.subheader("Category Prediction")
        st.write(f"##### Predicted Category: {predicted_category_name}")
        
        # Visualize category probabilities
        st.write(f"##### Category Probabilities")
        visualize_probabilities(probabilities[0], category_mapping)
        
        # Sentiment analysis results
        st.subheader("Sentiment Analysis")
        st.write(f"##### Sentiment: {sentiment_label}")
        # st.write(f"Sentiment probabilities: {sentiment_probs.tolist()}")

        # Visualize sentiment probabilities
        st.write(f"##### Sentiment Probabilities")
        visualize_sentiment_probabilities(sentiment_probs[0].detach().numpy(), sentiment_mapping)
    else:
        st.error("Please enter the event text for analysis.")
