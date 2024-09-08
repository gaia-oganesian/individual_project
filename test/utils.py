import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict


# Category mapping
category_mapping: Dict[int, str] = {
    0: 'Bounty',
    1: 'Bounty(LowQuality)',
    2: 'Other',
    3: 'Airdrop',
    4: 'Bounty, ICO'
}

# Dataset class
class NewTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, max_len: int = 512) -> None:
        """
        A PyTorch Dataset class for tokenizing input texts using a pre-trained tokenizer.

        Args:
            texts (List[str]): List of input texts.
            tokenizer (BertTokenizer): Pre-trained tokenizer to encode the texts.
            max_len (int): Maximum length for tokenization. Defaults to 512.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs = self.tokenizer(self.texts, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_len)
    
    def __len__(self) -> int:
        """ Returns the number of texts in the dataset. """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the tokenized input IDs and attention mask for the given index.

        Args:
            idx (int): Index of the text sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'input_ids' and 'attention_mask'.
        """
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Prediction function
def predict_category(model: BertForSequenceClassification, tokenizer: BertTokenizer, texts: List[str]) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Predicts the category for a list of texts using a pre-trained model and tokenizer.

    Args:
        model (BertForSequenceClassification): Pre-trained BERT model for classification.
        tokenizer (BertTokenizer): Tokenizer associated with the model.
        texts (List[str]): List of texts to classify.

    Returns:
        Tuple[List[int], List[torch.Tensor]]: A tuple containing the predicted categories and the list of probabilities.
    """
    dataset = NewTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predicted_categories = []
    probabilities_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_category = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            predicted_categories.append(predicted_category)
            probabilities_list.append(probabilities.cpu().numpy())

    return predicted_categories, probabilities_list

# Visualization function
def visualize_probabilities(probabilities: List[float], category_mapping: Dict[int, str]) -> None:
    """
    Visualizes the predicted category probabilities as a bar chart.

    Args:
        probabilities (List[float]): List of category probabilities.
        category_mapping (Dict[int, str]): Mapping of category indices to category names.
    """
    categories = [category_mapping[i] for i in range(len(probabilities))]
    probabilities_df = pd.DataFrame({'Category': categories, 'Probability': probabilities})
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Probability', y='Category', hue='Category', data=probabilities_df, palette="viridis", dodge=False, legend=False)
    
    # Add percentage labels
    for p in ax.patches:
        width = p.get_width()
        percentage = f'{width*100:.2f}%'
        x_offset = width + 0.01 if width < 0.03 else width - 0.02
        plt.text(x_offset, p.get_y() + p.get_height() / 2, percentage, ha='left' if width < 0.03 else 'center', va='center', color='black' if width < 0.03 else 'white')
    
    plt.xlabel('Probability')
    plt.ylabel('Category')
    plt.title('Predicted Probabilities by Category')
    plt.show()

# Model loading function
def load_model_and_tokenizer(model_path: str) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Loads the pre-trained model and tokenizer from the specified directory.

    Args:
        model_path (str): Path to the directory containing the pre-trained model and tokenizer.

    Returns:
        Tuple[BertForSequenceClassification, BertTokenizer]: Loaded model and tokenizer.
    """

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Ensure the model is in evaluation mode
    return model, tokenizer
