import os  
import html
import re
import string

from bs4 import BeautifulSoup  
from langdetect import detect, LangDetectException
import preprocessor as p
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from sklearn.utils import resample
from typing import List, Dict, Tuple 


# Function to ensure necessary NLTK resources are downloaded
def download_nltk_resources() -> None:
    """
    Downloads necessary NLTK resources if not already downloaded.
    """
    resources = ['punkt', 'stopwords', 'wordnet', 'words']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)


# Call to ensure all necessary NLTK data is downloaded
download_nltk_resources()

# Initialize NLTK tools
stop_words: set = set(stopwords.words('english'))
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
english_words: set = set(words.words())


# Function to clean HTML tags and decode HTML entities
def clean_html_and_decode(text: str) -> str:
    """
    Cleans HTML tags and decodes HTML entities in the text.

    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned text without HTML tags and decoded entities.
    """
    if isinstance(text, str):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = html.unescape(text)
        return text
    return ""


# Function to detect and clean non-English text
def process_text(text: str) -> str:
    """
    Detects and processes English text, removing non-English content, punctuation, digits, and stopwords.

    Args:
        text (str): Raw text to process.

    Returns:
        str: Cleaned and processed text.
    """
    if isinstance(text, str):
        try:
            if detect(text) != 'en':
                return ""  # Filter out non-English text
        except LangDetectException:
            return ""  # Filter out if language detection fails

        text = p.clean(text)  # Clean text using tweet-preprocessor
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.replace('<#NewLine>', '\n')  # Replace <#NewLine> with newline
        words_list = word_tokenize(text)  # Tokenize text
        words_list = [re.sub(r'[^\w\s]', '', word) for word in words_list]  # Remove punctuation
        words_list = [word for word in words_list if word not in stop_words]  # Remove stop words
        words_list = [lemmatizer.lemmatize(word) for word in words_list]  # Lemmatize words
        return ' '.join(words_list)
    return ""


# Function to preprocess text for embedding models
def preprocess_text_for_embeddings(text: str) -> str:
    """
    Preprocesses the text by lowercasing, removing punctuation, and lemmatizing.

    Args:
        text (str): Raw text to preprocess.

    Returns:
        str: Processed text suitable for embeddings.
    """
    if isinstance(text, str):
        text = text.lower()  # Lowercase text
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'<#NewLine>', ' ', text)  # Remove <#NewLine> tags
        text = re.sub(r'nan', ' ', text)  # Remove 'nan' values
        tokens = word_tokenize(text)  # Tokenize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and word in english_words]
        return ' '.join(tokens)  # Return processed text
    return ""


# Function to load and clean the dataset
def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Loads the dataset and applies initial cleaning steps.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    cleaned_events_df: pd.DataFrame = pd.read_csv(data_path, encoding='ISO-8859-1', encoding_errors='replace')
    cleaned_events_df['thread_id'] = cleaned_events_df['thread_id'].astype(str)
    cleaned_events_df = cleaned_events_df[cleaned_events_df['clean_title'].notna() & cleaned_events_df['clean_title'].str.strip().astype(bool)]
    cleaned_events_df['clean_title'] = cleaned_events_df['clean_title'].apply(clean_html_and_decode)
    cleaned_events_df = cleaned_events_df[cleaned_events_df['thread_id'].str.isdigit()].copy()
    cleaned_events_df = cleaned_events_df.reset_index(drop=True)
    return cleaned_events_df


# Function to filter and process data
def filter_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and processes the dataframe by applying text processing.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    df['processed_clean_title'] = df['clean_title'].apply(process_text)
    df['processed_post_tex'] = df['post_tex'].apply(process_text)

    df = df[(df['processed_clean_title'] != "") & (df['processed_post_tex'] != "")]
    df.drop_duplicates(inplace=True)
    return df


# Function to process the dataframe for clustering
def process_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the dataframe for clustering by tokenizing and embedding preparation.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe ready for clustering analysis.
    """
    df['clean_title'] = df['clean_title'].astype(str).fillna('')
    df['post_tex'] = df['post_tex'].astype(str).fillna('')
    df['general_rules'] = df['general_rules'].astype(str).fillna('')

    df['clean_title_tokens'] = df['clean_title'].apply(preprocess_text_for_embeddings)
    df['post_tex_tokens'] = df['post_tex'].apply(preprocess_text_for_embeddings)
    df['general_rules_tokens'] = df['general_rules'].apply(preprocess_text_for_embeddings)

    df['combined_text'] = df['clean_title_tokens'] + ' ' + df['post_tex_tokens']
    return df


# Function to filter top N categories and balance the dataset
def filter_top_categories_and_balance(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Selects the top N categories, then balances the dataset by resampling.

    Args:
        df (pd.DataFrame): Input dataframe.
        top_n (int): Number of top categories to select.

    Returns:
        pd.DataFrame: Balanced dataframe with top categories.
    """
    # Select top N categories
    top_categories = df['categories'].value_counts().nlargest(top_n).index
    filtered_df: pd.DataFrame = df[df['categories'].isin(top_categories)]

    # Convert categories to numerical labels
    category_mapping: Dict[str, int] = {category: idx for idx, category in enumerate(top_categories)}
    filtered_df['category_label'] = filtered_df['categories'].map(category_mapping)

    # Balance the dataset by resampling
    dfs: List[pd.DataFrame] = [resample(filtered_df[filtered_df['category_label'] == label], 
                                        replace=True,  # Sample with replacement
                                        n_samples=filtered_df['category_label'].value_counts().min(),  # Match the smallest class
                                        random_state=42) for label in filtered_df['category_label'].unique()]
    
    balanced_df: pd.DataFrame = pd.concat(dfs)
    
    return balanced_df


# Function to save the cleaned dataframe
def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned dataframe to a CSV file.

    Args:
        df (pd.DataFrame): Dataframe to save.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"The cleaned data has been saved to {output_path}")