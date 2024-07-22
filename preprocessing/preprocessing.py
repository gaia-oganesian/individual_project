

# import pandas as pd
# from bs4 import BeautifulSoup
# import html
# import re
# from langdetect import detect, LangDetectException
# import preprocessor as p
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords, words
# from nltk.stem import WordNetLemmatizer
# import nltk
# import string

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('words')

# # Initialize NLTK tools
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# english_words = set(words.words())

# # Function to clean HTML tags and decode HTML entities
# def clean_html_and_decode(text):
#     """
#     Remove HTML tags and decode HTML entities from a given text.
    
#     Args:
#     text (str): Text to be cleaned.
    
#     Returns:
#     str: Cleaned text.
#     """
#     text = BeautifulSoup(text, "html.parser").get_text()
#     text = html.unescape(text)
#     return text

# # Function to clean and process text
# def process_text(text):
#     """
#     Clean and process the text by detecting language, removing digits,
#     punctuation, stop words, and applying lemmatization.
    
#     Args:
#     text (str): Text to be processed.
    
#     Returns:
#     str: Processed text or an empty string if not valid.
#     """
#     if isinstance(text, str):
#         try:
#             if detect(text) != 'en':
#                 return ""  # Filter out non-English text
#         except LangDetectException:
#             return ""  # Filter out if language detection fails

#         text = p.clean(text)  # Clean text using tweet-preprocessor
#         text = re.sub(r'\d+', '', text)  # Remove digits
#         text = text.replace('<#NewLine>', '\n')  # Replace <#NewLine> with newline
#         words = word_tokenize(text)  # Tokenize text
#         words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove punctuation
#         words = [word for word in words if word not in stop_words]  # Remove stop words
#         words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
#         return ' '.join(words)
#     else:
#         return ""

# # Function to preprocess text for embeddings
# def preprocess_text_for_embeddings(text):
#     text = text.lower()  # Lowercase
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = re.sub(r'<#NewLine>', ' ', text)  # Remove <#NewLine> tags
#     text = re.sub(r'nan', ' ', text)  # Remove 'nan' values
#     tokens = word_tokenize(text)  # Tokenize
#     tokens = [word for word in tokens if word not in stop_words and word in english_words]  # Remove stop words and non-English words
#     return ' '.join(tokens)

# # Function to load and clean data
# def load_and_clean_data(data_path):
#     """
#     Load data from a CSV file and perform cleaning steps.
    
#     Args:
#     data_path (str): Path to the CSV file.
    
#     Returns:
#     pd.DataFrame: Cleaned DataFrame.
#     """
#     cleaned_events_df = pd.read_csv(data_path, encoding='ISO-8859-1', encoding_errors='replace')
#     cleaned_events_df['thread_id'] = cleaned_events_df['thread_id'].astype(str)
#     cleaned_events_df = cleaned_events_df[cleaned_events_df['clean_title'].notna() & cleaned_events_df['clean_title'].str.strip().astype(bool)]
#     cleaned_events_df['clean_title'] = cleaned_events_df['clean_title'].apply(clean_html_and_decode)
#     cleaned_events_df = cleaned_events_df[cleaned_events_df['thread_id'].str.isdigit()].copy()
#     cleaned_events_df = cleaned_events_df.reset_index(drop=True)
#     return cleaned_events_df

# # Function to filter and process data
# def filter_and_process_data(df):
#     """
#     Filter out empty processed texts and remove duplicates from the DataFrame.
    
#     Args:
#     df (pd.DataFrame): DataFrame to be filtered and processed.
    
#     Returns:
#     pd.DataFrame: Processed DataFrame.
#     """
#     df['processed_post_tex'] = df['post_tex'].apply(process_text)
#     df = df[df['processed_post_tex'] != ""]
#     df.drop_duplicates(inplace=True)
#     return df

# # Function to process data for clustering
# def process_for_clustering(df):
#     df['clean_title_tokens'] = df['clean_title'].apply(preprocess_text_for_embeddings)
#     df['post_tex_tokens'] = df['post_tex'].apply(preprocess_text_for_embeddings)
#     df['combined_text'] = df['clean_title_tokens'] + ' ' + df['post_tex_tokens']
#     return df

# # Function to save cleaned data
# def save_cleaned_data(df, output_path):
#     """
#     Save the cleaned DataFrame to a CSV file.
    
#     Args:
#     df (pd.DataFrame): DataFrame to be saved.
#     output_path (str): Path to the output CSV file.
#     """
#     df.to_csv(output_path, index=False)
#     print(f"The cleaned data have been saved to {output_path}")



import pandas as pd
from bs4 import BeautifulSoup
import html
import re
from langdetect import detect, LangDetectException
import preprocessor as p
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
english_words = set(words.words())

# Function to clean HTML tags and decode HTML entities
def clean_html_and_decode(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    return text

# Function to clean and process text
def process_text(text):
    if isinstance(text, str):
        try:
            if detect(text) != 'en':
                return ""  # Filter out non-English text
        except LangDetectException:
            return ""  # Filter out if language detection fails

        text = p.clean(text)  # Clean text using tweet-preprocessor
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.replace('<#NewLine>', '\n')  # Replace <#NewLine> with newline
        words = word_tokenize(text)  # Tokenize text
        words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove punctuation
        words = [word for word in words if word not in stop_words]  # Remove stop words
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        return ' '.join(words)
    else:
        return ""

# Function to preprocess text for embeddings
def preprocess_text_for_embeddings(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'<#NewLine>', ' ', text)  # Remove <#NewLine> tags
    text = re.sub(r'nan', ' ', text)  # Remove 'nan' values
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and word in english_words]  # Lemmatize and remove stop words
    return ' '.join(tokens)  # Return tokens as string

def load_and_clean_data(data_path):
    cleaned_events_df = pd.read_csv(data_path, encoding='ISO-8859-1', encoding_errors='replace')
    cleaned_events_df['thread_id'] = cleaned_events_df['thread_id'].astype(str)
    cleaned_events_df = cleaned_events_df[cleaned_events_df['clean_title'].notna() & cleaned_events_df['clean_title'].str.strip().astype(bool)]
    cleaned_events_df['clean_title'] = cleaned_events_df['clean_title'].apply(clean_html_and_decode)
    cleaned_events_df = cleaned_events_df[cleaned_events_df['thread_id'].str.isdigit()].copy()
    cleaned_events_df = cleaned_events_df.reset_index(drop=True)
    return cleaned_events_df

def filter_and_process_data(df):
    df['processed_clean_title'] = df['clean_title'].apply(process_text)
    df['processed_post_tex'] = df['post_tex'].apply(process_text)
    # df['processed_general_rules'] = df['general_rules'].apply(process_text)

    df = df[(df['processed_clean_title'] != "") & (df['processed_post_tex'] != "")]
    df.drop_duplicates(inplace=True)
    return df

def process_for_clustering(df):
    # Ensure columns are treated as strings and handle NaNs
    df['clean_title'] = df['clean_title'].astype(str).fillna('')
    df['post_tex'] = df['post_tex'].astype(str).fillna('')
    df['general_rules'] = df['general_rules'].astype(str).fillna('')

    # Apply text preprocessing for embeddings
    df['clean_title_tokens'] = df['clean_title'].apply(preprocess_text_for_embeddings)
    df['post_tex_tokens'] = df['post_tex'].apply(preprocess_text_for_embeddings)
    df['general_rules_tokens'] = df['general_rules'].apply(preprocess_text_for_embeddings)
    
    # Combine the processed text columns for embeddings
    df['combined_text'] = df['clean_title_tokens'] + ' ' + df['post_tex_tokens'] 
    return df


def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"The cleaned data have been saved to {output_path}")
