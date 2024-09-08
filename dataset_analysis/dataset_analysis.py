import re  
from collections import Counter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
from nltk import bigrams, trigrams
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk  

# Ensure necessary NLTK resources are downloaded
def download_nltk_resources() -> None:
    """
    Ensure that necessary NLTK resources are downloaded.
    """
    resources = ['punkt', 'stopwords', 'words']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)


# Load data from CSV
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from the provided CSV file path.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    return pd.read_csv(data_path, encoding='ISO-8859-1', encoding_errors='replace')

# Preprocess text data
def preprocess_text(text: str) -> List[str]:
    """
    Preprocess the input text by lowercasing, removing punctuation, numbers, and stopwords.
    
    Args:
        text (str): Input text to be preprocessed.
    
    Returns:
        List[str]: List of preprocessed tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'<#NewLine>', ' ', text)  # Remove <#NewLine> tags
    tokens = word_tokenize(text)
    english_words = set(words.words())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word in english_words]
    return tokens

# Common plotting function for bar charts
def plot_bar_chart(data: List[int], x_labels: List[str], title: str, xlabel: str, ylabel: str, rotation: int = 45, color: str = '#339fff') -> None:
    """
    Plot a bar chart with the given data and labels.

    Args:
        data (List[int]): Data to plot.
        x_labels (List[str]): Labels for the x-axis.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        rotation (int): Rotation for x-axis labels.
        color (str): Bar color.
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_labels, data, color=color, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Common plotting function for bar charts with percentage formatting
def plot_bar_chart_percent(data: List[float], x_labels: List[str], title: str, xlabel: str, ylabel: str, rotation: int = 45, color: str = '#339fff') -> None:
    """
    Plot a bar chart with percentage formatting on the data.

    Args:
        data (List[float]): Data to plot as percentages.
        x_labels (List[str]): Labels for the x-axis.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        rotation (int): Rotation for x-axis labels.
        color (str): Bar color.
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_labels, data, color=color, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to filter out invalid categories
def filter_valid_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out events that contain the words 'closed' or 'nan' in their categories.

    Args:
        df (pd.DataFrame): DataFrame containing the 'categories' column.

    Returns:
        pd.DataFrame: Filtered DataFrame without 'closed' or 'nan' categories.
    """
    return df[~df['categories'].str.contains('closed|nan', case=False, na=False)]

# Count social media URLs in the dataset
def count_social_media_urls(df: pd.DataFrame, social_media_keywords: Dict[str, str]) -> Dict[str, int]:
    """
    Count occurrences of social media keywords in the 'social_media_urls' column.

    Args:
        df (pd.DataFrame): DataFrame containing 'social_media_urls' column.
        social_media_keywords (Dict[str, str]): Dictionary of social media keywords.

    Returns:
        Dict[str, int]: Dictionary with counts of social media keywords.
    """
    social_media_counts = {key: 0 for key in social_media_keywords}
    social_media_counts['Other'] = 0

    for urls in df['social_media_urls']:
        if urls.lower() == 'nan':
            continue
        found = False
        for key, keyword in social_media_keywords.items():
            if keyword in urls.lower():
                social_media_counts[key] += 1
                found = True
                break
        if not found:
            social_media_counts['Other'] += 1

    return social_media_counts

# Extract percentage from text
def extract_percentage(text: str) -> List[int]:
    """
    Extract percentage values from text using regex.

    Args:
        text (str): Text to extract percentages from.

    Returns:
        List[int]: List of integer percentage values.
    """
    matches = re.findall(r'(\d+)%', text)
    return [int(match) for match in matches]

# Count reward allocation sums from text
def count_reward_allocations(df: pd.DataFrame, reward_keywords: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Count occurrences of reward keywords and sum their percentages.

    Args:
        df (pd.DataFrame): DataFrame containing the 'reward_allocation' column.
        reward_keywords (List[str]): List of reward keywords to look for.

    Returns:
        Tuple[Dict[str, int], List[str]]: Dictionary of summed percentages and a list of 'Other' keywords.
    """
    reward_allocation_sums = {key: 0 for key in reward_keywords}
    reward_allocation_sums['Other'] = 0
    other_keywords_list = []

    for allocation in df['reward_allocation']:
        percentages = extract_percentage(allocation)

        found_keywords = set()
        for keyword in reward_keywords:
            if keyword in allocation:
                found_keywords.add(keyword)
                keyword_matches = re.findall(rf'{keyword}', allocation)
                for match in keyword_matches:
                    index = allocation.find(match) + len(match)
                    if index < len(allocation):
                        match_percentages = re.findall(r'(\d+)%', allocation[index:])
                        for percent in match_percentages:
                            reward_allocation_sums[keyword] += int(percent)

        # Handle 'Other' category
        other_text = re.sub(r'(\d+)%', '', allocation)
        other_keywords = set(re.findall(r'\b[A-Z]+\b', other_text)) - set(reward_keywords)
        for keyword in other_keywords:
            other_keywords_list.append(keyword)
            reward_allocation_sums['Other'] += 1 / len(other_keywords) * sum(percentages)

    return reward_allocation_sums, other_keywords_list

# Calculate percentage share
def calculate_share(sums: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate percentage share for each category in the sum.

    Args:
        sums (Dict[str, int]): Dictionary of category sums.

    Returns:
        Dict[str, float]: Dictionary with percentage shares for each category.
    """
    total_percentage = sum(sums.values())
    if total_percentage == 0:
        return {key: 0 for key in sums.keys()}
    return {key: (value / total_percentage) * 100 for key, value in sums.items()}

# Plot frequency distribution for words, bigrams, trigrams
def plot_freq_distribution(ax: plt.Axes, freq_dist: List[Tuple[str, int]], title: str) -> None:
    """
    Plot the frequency distribution of words, bigrams, or trigrams.

    Args:
        ax (plt.Axes): Matplotlib Axes object.
        freq_dist (List[Tuple[str, int]]): Frequency distribution of terms.
        title (str): Title for the plot.
    """
    labels, values = zip(*freq_dist)
    labels = [' '.join(label) if isinstance(label, tuple) else label for label in labels]
    ax.barh(labels, values, color=plt.cm.Paired(range(len(freq_dist))))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Terms')

# Perform TF-IDF analysis
def tfidf_analysis(events_df: pd.DataFrame, top_categories: List[str], data_columns: List[str], category_columns: str) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """
    Perform TF-IDF analysis for the provided categories and columns.

    Args:
        events_df (pd.DataFrame): DataFrame containing event data.
        top_categories (List[str]): List of top categories for analysis.
        data_columns (List[str]): List of columns to analyze.
        category_columns (str): Column name for categories.

    Returns:
        Dict[Tuple[str, str], List[Tuple[str, float]]]: Results of the TF-IDF analysis.
    """
    results = {}
    for category in top_categories:
        category_df = events_df[events_df[category_columns] == category]
        for column in data_columns:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(category_df[column])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            top_tfidf_features = get_top_tfidf_features(tfidf_matrix, feature_names)
            results[(category, column)] = top_tfidf_features
            print(f'TF-IDF Analysis for Category: {category}, Column: {column}')
            print(pd.DataFrame(top_tfidf_features, columns=['Term', 'TF-IDF Score']).head(10))
    return results

# Plot TF-IDF analysis results
def plot_tfidf_results(results: Dict[Tuple[str, str], List[Tuple[str, float]]]) -> None:
    """
    Plot the results of TF-IDF analysis.

    Args:
        results (Dict[Tuple[str, str], List[Tuple[str, float]]]): Results of the TF-IDF analysis.
    """
    for (category, column), tfidf_features in results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle(f'Category: {category}', fontsize=16)
        plot_tfidf_distribution(ax, tfidf_features, f'Top TF-IDF Features in {column.replace("_tokens", "").title()}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Perform frequency analysis
def frequency_analysis(events_df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]]:
    """
    Perform frequency analysis for words, bigrams, and trigrams.

    Args:
        events_df (pd.DataFrame): DataFrame containing event data.
        columns (List[str]): List of columns to analyze.

    Returns:
        Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]]: Results of frequency analysis.
    """
    freq_results = {}
    for column in columns:
        all_tokens = [token for sublist in events_df[column] for token in sublist]
        word_freq = Counter(all_tokens).most_common(20)
        bigram_freq = Counter([bigram for sublist in events_df[column] for bigram in bigrams(sublist)]).most_common(20)
        trigram_freq = Counter([trigram for sublist in events_df[column] for trigram in trigrams(sublist)]).most_common(20)
        freq_results[column] = (word_freq, bigram_freq, trigram_freq)
        print(f"Frequency Analysis for Column: {column}")
        print(f"Most Common Words:\n{word_freq}")
        print(f"Most Common Bigrams:\n{bigram_freq}")
        print(f"Most Common Trigrams:\n{trigram_freq}")
    return freq_results

# Plot frequency analysis results
def plot_frequency_results(freq_results: Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]]) -> None:
    """
    Plot the results of frequency analysis for words, bigrams, and trigrams.

    Args:
        freq_results (Dict[str, Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]]): Results of frequency analysis.
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 24))
    columns = list(freq_results.keys())
    for i, column in enumerate(columns):
        plot_freq_distribution(axes[0, i], freq_results[column][0], f'Most Common Words in {column.replace("_tokens", "").title()}')
        plot_freq_distribution(axes[1, i], freq_results[column][1], f'Most Common Bigrams in {column.replace("_tokens", "").title()}')
        plot_freq_distribution(axes[2, i], freq_results[column][2], f'Most Common Trigrams in {column.replace("_tokens", "").title()}')
    plt.tight_layout()
    plt.show()

# Get top TF-IDF features
def get_top_tfidf_features(tfidf_matrix, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Get the top TF-IDF features from the matrix.

    Args:
        tfidf_matrix: TF-IDF matrix.
        feature_names (List[str]): List of feature names.
        top_n (int): Number of top features to return.

    Returns:
        List[Tuple[str, float]]: List of top TF-IDF features with scores.
    """
    sorted_indices = tfidf_matrix.sum(axis=0).A1.argsort()[-top_n:][::-1]
    return [(feature_names[i], tfidf_matrix[:, i].sum()) for i in sorted_indices]

# Plot TF-IDF distribution
def plot_tfidf_distribution(ax: plt.Axes, tfidf_features: List[Tuple[str, float]], title: str) -> None:
    """
    Plot the TF-IDF score distribution.

    Args:
        ax (plt.Axes): Matplotlib Axes object.
        tfidf_features (List[Tuple[str, float]]): List of TF-IDF features with scores.
        title (str): Title for the plot.
    """
    labels, values = zip(*tfidf_features)
    ax.barh(labels, values, color=plt.cm.Paired(range(len(tfidf_features))))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('TF-IDF Score')
    ax.set_ylabel('Terms')
