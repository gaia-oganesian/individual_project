# Inside Coordinated Cryptocurrency-Related Social Media Campaigns

## Topic Overview
This project aims to uncover manipulation strategies in cryptocurrency markets, which are heavily influenced by coordinated social media campaigns. Using a dataset of cryptocurrency-related bounty events spanning from May 2014 to December 2022, this research focuses on analysing these campaigns' structure, categorising their content, and evaluating the sentiment associated with them. The key objective is to develop an event categorisation and sentiment analysis app that provides actionable insights into how these campaigns may impact market behavior.

## Dataset
The primary dataset used in this project can be downloaded from Zenodo:

- [A Dataset of Coordinated Cryptocurrency-Related Social Media Campaigns](https://zenodo.org/records/7813450)

The dataset includes detailed records of coordinated campaigns, such as airdrops and bounties, on social media, specifically focusing on how these campaigns were designed to manipulate market sentiment.



## Key Contributions
- **Dataset Analysis**: Applied clustering techniques to discover manipulation tactics in cryptocurrency markets. Performed textual analysis to uncover common strategies used in coordinated social media campaigns.

- **Category Classification**: Fine-tuned various models, including BERT, FinBERT, CryptoBERT, and OpenAI Embedding models, to classify text data into key categories. Achieved the highest accuracy using the fine-tuned BERT model, demonstrating the importance of domain-specific training for accurate classification.

- **Sentiment Analysis**: Utilised pre-trained models like FinBERT and CryptoBERT to perform sentiment analysis on cryptocurrency-related social media content.
CryptoBERT proved to be particularly effective in capturing real-time market sentiment.

- **Event Categorisation and Sentiment Analysis App**: Developed a web application that combines classification and sentiment analysis, allowing users to predict the categories and sentiment of cryptocurrency-related events.


## Installation

### Prerequisites

Make sure you have the following installed:

- **Python 3.8+**
- **PyTorch** 
- **Streamlit** (for running the web application)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://gitlab.com/gaiaoganesian/individual_project.git
   cd individual_project

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

3. **Download the dataset**: Bitcointalk Bounties(Altcoins) Dataset.zip -> events.tsv

4. **API Keys**: For running the code, you need to have an **OpenAI API Key** and a **Hugging Face API Token**.
   
   - Rename `.env_example` to `.env`.
   - Add your OpenAI and Hugging Face tokens to the `.env` file:
     ```
     OPENAI_API_KEY=<your_openai_api_key>
     HUGGING_FACE_API_TOKEN=<your_hugging_face_api_token>

5. **Run the Streamlit App**:
    ```bash
    streamlit run app.py

6. **Running Unit Tests**:
    ```bash
    pytest test/unit_test.py

### Note:
All models in the `models/` folder were trained and evaluated using **GPU** resources, primarily through **Google Colab** to take advantage of its hardware acceleration.


## File Structure
```bash

INDIVIDUAL_PROJECT/
├── dataset/                           # Cleaned and balanced datasets
│   ├── cleaned_events.csv             # Cleaned dataset of cryptocurrency campaigns
│   └── balanced_dataset.csv           # Cleaned and balanced dataset for analysis
│
├── embeddings/                        # Embedding models and utilities
│   ├── base.py                        # Base embedding class
│   ├── huggingface.py                 # Hugging Face embeddings integration
│   ├── openai.py                      # OpenAI embedding models integration
│   └── __init__.py
│
├── preprocessing/                     # Data preprocessing scripts
│   ├── preprocessing.py               # Preprocessing script for tokenising and cleaning data
│   ├── preprocessing_demo.ipynb       # Demo notebook for data preprocessing
│   └── __init__.py
│
├── dataset_analysis/                  # Dataset analysis notebooks and scripts
│   ├── dataset_analysis_demo.ipynb    # Jupyter notebook demo of dataset analysis
│   ├── dataset_analysis.py            # Python script for dataset analysis
│   └── __init__.py
│
├── clustering/                        # Clustering and analysis notebooks
│   ├── clustering_demo.ipynb          # Demo of clustering techniques 
│   └── __init__.py
│
├── models/                            # Model training and analysis scripts
│   ├── classification/                # Classification models and utilities
│   │   ├── bert_adam.ipynb            # BERT model trained with Adam optimizer
│   │   ├── bert_adamw.ipynb           # BERT model trained with AdamW optimizer
│   │   ├── bert_large_ft.ipynb        # Fine-tuned large BERT model
│   │   ├── cryptobert.ipynb           # CryptoBERT model for classification analysis
│   │   ├── finbert.ipynb              # FinBERT model for classification analysis
│   │   ├── utils.py                   # Helper functions for classification models
│   │
│   ├── prediction/                    # Prediction scripts for category and sentiment
│   │   ├── cryptobert_predicting.ipynb # CryptoBERT sentiment prediction script
│   │   ├── finbert_predicting.ipynb   # FinBERT sentiment prediction script
│   │   └── utils.py                   # Helper functions for predictions
│   │
│   ├── general_rules/                 # GPT-based rule extraction and summarisation
│   │   └── gpt_mini.ipynb             # Demo of GPT (gpt-4-mini) rule extraction
│   │
│   └── __init__.py
│
├── test/                              # Unit tests and app testing scripts
│   ├── unit_test.py                   # Unit tests for classification and sentiment models
│   ├── demo.ipynb                     # Demo for testing models on small samples
│   ├── app.py                         # Streamlit web app for event categorisation and sentiment analysis
│   └── __init__.py
│
├── .env                               # Environment variables (API keys, etc.)
├── models_comparison.xlsx             # Excel file comparing model performances
├── README.md                          # Project documentation
├── requirements.txt                   # Project dependencies
└── trained_models/                    # Directory to store pre-trained models
