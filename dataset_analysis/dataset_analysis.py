import pandas as pd
from collections import Counter
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
english_words = set(words.words())

def load_data(data_path):
    events_df = pd.read_csv(data_path, encoding='ISO-8859-1', encoding_errors='replace')
    return events_df

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'<#NewLine>', ' ', text)
    text = re.sub(r'nan', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word in english_words]
    return ' '.join(tokens)



# Function to plot frequency distribution
def plot_freq_distribution(ax, freq_dist, title):
    labels, values = zip(*freq_dist)
    labels = [' '.join(label) if isinstance(label, tuple) else label for label in labels]
    ax.barh(labels, values, color=plt.cm.Paired(range(len(freq_dist))))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Terms')


def tfidf_analysis(events_df, top_categories, data_columns, category_columns):
    results = {}
    
    for category in top_categories:
        category_df = events_df[events_df[category_columns] == category]
        
        for column in data_columns:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(category_df[column])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            top_tfidf_features = get_top_tfidf_features(tfidf_matrix, feature_names)
            
            results[(category, column)] = top_tfidf_features
            
    return results

def plot_tfidf_results(results):
    for (category, column), tfidf_features in results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle(f'Category: {category}', fontsize=16)
        plot_tfidf_distribution(ax, tfidf_features, f'Top TF-IDF Features in {column.replace("_tokens", "").title()}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
def frequency_analysis(events_df, columns):
    freq_results = {}
    
    for column in columns:
        all_tokens = [token for sublist in events_df[column] for token in sublist]
        word_freq = Counter(all_tokens)
        most_common_words = word_freq.most_common(20)
        
        all_bigrams = [bigram for sublist in events_df[column] for bigram in bigrams(sublist)]
        bigram_freq = Counter(all_bigrams)
        most_common_bigrams = bigram_freq.most_common(20)
        
        all_trigrams = [trigram for sublist in events_df[column] for trigram in trigrams(sublist)]
        trigram_freq = Counter(all_trigrams)
        most_common_trigrams = trigram_freq.most_common(20)
        
        freq_results[column] = (most_common_words, most_common_bigrams, most_common_trigrams)
    
    return freq_results

def plot_frequency_results(freq_results):
    fig, axes = plt.subplots(3, 3, figsize=(20, 24))
    
    columns = list(freq_results.keys())
    
    for i, column in enumerate(columns):
        plot_freq_distribution(axes[0, i], freq_results[column][0], f'Most Common Words in {column.replace("_tokens", "").title()}')
        plot_freq_distribution(axes[1, i], freq_results[column][1], f'Most Common Bigrams in {column.replace("_tokens", "").title()}')
        plot_freq_distribution(axes[2, i], freq_results[column][2], f'Most Common Trigrams in {column.replace("_tokens", "").title()}')
    
    plt.tight_layout()
    plt.show()
