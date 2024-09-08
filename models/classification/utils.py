import os
import nltk
from nltk.corpus import wordnet
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics.functional import accuracy
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.model_selection import train_test_split


nltk.download('wordnet')

def load_or_split_dataset(input_file, train_file, validation_file, test_file, test_size=0.2, random_state=42):
    """
    Load existing train, validation, and test splits if they exist.
    If they don't exist, split the input dataset and save the splits.
    
    Args:
        input_file (str): Path to the original dataset (CSV file).
        train_file (str): Path to save the training set.
        validation_file (str): Path to save the validation set.
        test_file (str): Path to save the test set.
        test_size (float): The proportion of the dataset to include in the test and validation sets.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: DataFrames for train, validation, and test sets.
    """
    if os.path.exists(train_file) and os.path.exists(validation_file) and os.path.exists(test_file):
        # Load existing splits
        train_df = pd.read_csv(train_file)
        validation_df = pd.read_csv(validation_file)
        test_df = pd.read_csv(test_file)
        print("Loaded pre-existing train, validation, and test sets.")
    else:
        # Load the original dataset
        df = pd.read_csv(input_file)
        print("Splitting dataset into train, validation, and test sets.")

        # Split into train, validation, and test sets (80% train, 10% validation, 10% test)
        train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

        # Save the splits
        train_df.to_csv(train_file, index=False)
        validation_df.to_csv(validation_file, index=False)
        test_df.to_csv(test_file, index=False)
        print("Saved train, validation, and test sets.")
    
    return train_df, validation_df, test_df



class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_len: int) -> None:
        """
        Initializes the TextDataset for tokenizing and encoding text data.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data (text and labels).
            tokenizer (BertTokenizer): Tokenizer to convert text to BERT tokens.
            max_len (int): Maximum length of the input sequence for BERT model.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single data sample as a dictionary.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels as tensors.
        """
        row = self.dataframe.iloc[idx]
        text = row['combined_text']
        label = row['category_label']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

class TextDataModule(LightningDataModule):
    def __init__(self, train_file: str, validation_file: str, test_file: str, tokenizer_name: str, batch_size: int = 32, max_len: int = 128) -> None:
        """
        Initializes the DataModule for loading the training, validation, and test datasets.
        
        Args:
            train_file (str): Path to the training dataset CSV.
            validation_file (str): Path to the validation dataset CSV.
            test_file (str): Path to the test dataset CSV.
            tokenizer_name (str): Name of the BERT tokenizer to use.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            max_len (int, optional): Maximum length of the input sequences. Defaults to 128.
        """
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the datasets for training, validation, and testing by loading the CSV files.

        Args:
            stage (Optional[str]): The stage of the process (train, validation, test). Defaults to None.
        """
        self.train_df = pd.read_csv(self.train_file)
        self.val_df = pd.read_csv(self.validation_file)
        self.test_df = pd.read_csv(self.test_file)

        self.train_dataset = TextDataset(self.train_df, self.tokenizer, self.max_len)
        self.val_dataset = TextDataset(self.val_df, self.tokenizer, self.max_len)
        self.test_dataset = TextDataset(self.test_df, self.tokenizer, self.max_len)

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training dataset.

        Returns:
            DataLoader: Dataloader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the validation dataset.

        Returns:
            DataLoader: Dataloader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the test dataset.

        Returns:
            DataLoader: Dataloader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class BertClassifier(LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate: float, optimizer: str = 'Adam') -> None:
        """
        Initializes the BERT classifier for text classification.
        
        Args:
            model_name (str): Name of the pre-trained BERT model.
            num_classes (int): Number of output classes for classification.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str, optional): Optimizer to use ('Adam' or 'AdamW'). Defaults to 'Adam'.
        """
        super(BertClassifier, self).__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.test_preds: List[int] = []
        self.test_labels: List[int] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.
        
        Args:
            input_ids (torch.Tensor): Tensor of input IDs from the tokenizer.
            attention_mask (torch.Tensor): Attention mask to avoid attending to padding tokens.

        Returns:
            torch.Tensor: Logits for each class.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step during each batch of training.
        
        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step during each batch of validation.

        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Test step during each batch of testing.

        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        if self.hparams.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'AdamW':
            return AdamW(self.parameters(), lr=self.hparams.learning_rate)
    
    def save_model(self, directory: str) -> None:
        """
        Saves the model and tokenizer to the specified directory.

        Args:
            directory (str): Directory to save the model and tokenizer.
        """
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the testing epoch to compute metrics and visualize the confusion matrix.
        """
        precision = precision_score(self.test_labels, self.test_preds, average='weighted')
        recall = recall_score(self.test_labels, self.test_preds, average='weighted')
        f1 = f1_score(self.test_labels, self.test_preds, average='weighted')
        cm = confusion_matrix(self.test_labels, self.test_preds)

        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

##### Fine-tuning #####
class TextAugmentor:
    def __init__(self, synonym_replacement_n: int = 3, random_swap_n: int = 3) -> None:
        """
        Initialize the TextAugmentor class with specific parameters for augmentation techniques.
        
        Args:
            synonym_replacement_n (int): Number of words to replace with synonyms during synonym replacement.
            random_swap_n (int): Number of word pairs to swap during random swapping.
        """
        self.synonym_replacement_n = synonym_replacement_n
        self.random_swap_n = random_swap_n

    def synonym_replacement(self, text: str) -> str:
        """
        Perform synonym replacement on the input text.
        
        Args:
            text (str): The input text to augment.
        
        Returns:
            str: The augmented text with synonyms replaced.
        """
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
        random.shuffle(random_word_list)
        num_replacements = min(self.synonym_replacement_n, len(random_word_list))

        for _ in range(num_replacements):
            word_to_replace = random_word_list.pop()
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                new_words = [synonym if word == word_to_replace else word for word in new_words]

        return ' '.join(new_words)

    def random_swap(self, text: str) -> str:
        """
        Perform random word swapping on the input text.
        
        Args:
            text (str): The input text to augment.
        
        Returns:
            str: The augmented text with random word swaps.
        """
        words = text.split()
        if len(words) < 2:
            return text
        new_words = words.copy()
        num_swaps = min(self.random_swap_n, len(words) // 2)
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return ' '.join(new_words)

    def augment_text(self, text: str, techniques=None) -> str:
        """
        Augment the input text using specified augmentation techniques.
        
        Args:
            text (str): The input text to augment.
            techniques (list of str, optional): List of augmentation techniques to apply. 
                Default techniques are synonym replacement and random swapping.
        
        Returns:
            str: The augmented text.
        """
        if techniques is None:
            techniques = ['synonym_replacement', 'random_swap']

        for technique in techniques:
            if technique == 'synonym_replacement':
                text = self.synonym_replacement(text)
            elif technique == 'random_swap':
                text = self.random_swap(text)

        return text
    

class TextDataModule_ft(LightningDataModule):
    def __init__(
        self,
        train_file: str,
        validation_file: str,
        test_file: str,
        tokenizer_name: str,
        batch_size: int = 32,
        max_len: int = 128,
        augment: bool = False,
        num_workers: int = 4
    ) -> None:
        """
        Initialize the TextDataModule with training, validation, and test file paths.
        
        Args:
            train_file (str): Path to the training dataset CSV.
            validation_file (str): Path to the validation dataset CSV.
            test_file (str): Path to the test dataset CSV.
            tokenizer_name (str): Name of the BERT tokenizer to use.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            max_len (int, optional): Maximum length of the input sequences. Defaults to 128.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            num_workers (int, optional): Number of worker threads for DataLoader. Defaults to 4.
        """
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_len = max_len
        self.augment = augment
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the datasets for training, validation, and testing by loading the CSV files.

        Args:
            stage (Optional[str]): The stage of the process (train, validation, test). Defaults to None.
        """
        self.train_df = pd.read_csv(self.train_file)
        self.val_df = pd.read_csv(self.validation_file)
        self.test_df = pd.read_csv(self.test_file)
        self.train_dataset = TextDataset(self.train_df, self.tokenizer, self.max_len)
        self.val_dataset = TextDataset(self.val_df, self.tokenizer, self.max_len)
        self.test_dataset = TextDataset(self.test_df, self.tokenizer, self.max_len)

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training dataset.

        Returns:
            DataLoader: Dataloader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the validation dataset.

        Returns:
            DataLoader: Dataloader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the test dataset.

        Returns:
            DataLoader: Dataloader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

class BertClassifier_ft(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float,
        max_len: int,
        weight_decay: float,
        augmentor: Optional[TextAugmentor] = None
    ) -> None:
        """
        Initialize the fine-tuned BERT classifier for text classification.
        
        Args:
            model_name (str): Name of the pre-trained BERT model.
            num_classes (int): Number of output classes for classification.
            learning_rate (float): Learning rate for the optimizer.
            max_len (int): Maximum input length for the tokenizer.
            weight_decay (float): Weight decay for the optimizer.
            augmentor (Optional[TextAugmentor]): An instance of TextAugmentor for data augmentation. Defaults to None.
        """
        super(BertClassifier_ft, self).__init__()
        self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.weight_decay = weight_decay
        self.augmentor = augmentor

        self.test_preds = []
        self.test_labels = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.
        
        Args:
            input_ids (torch.Tensor): Tensor of input IDs from the tokenizer.
            attention_mask (torch.Tensor): Attention mask to avoid attending to padding tokens.

        Returns:
            torch.Tensor: Logits for each class.
        """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step during each batch of training, including optional text augmentation.
        
        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Augment text data
        if self.augmentor:
            augmented_input_ids = []
            augmented_attention_mask = []
            for i in range(len(input_ids)):
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                augmented_text = self.augmentor.augment_text(text)  
                augmented_tokens = self.tokenizer.encode_plus(
                    augmented_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.hparams.max_len,
                    return_tensors='pt'
                )
                augmented_input_ids.append(augmented_tokens['input_ids'].squeeze(0))
                augmented_attention_mask.append(augmented_tokens['attention_mask'].squeeze(0))

            input_ids = torch.stack(augmented_input_ids).to(self.device)
            attention_mask = torch.stack(augmented_attention_mask).to(self.device)

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss


    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step during each batch of validation.

        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Test step during each batch of testing.

        Args:
            batch (dict): The batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Actions to perform after the test epoch ends, such as calculating precision, recall, F1, and plotting confusion matrix.
        """
        precision = precision_score(self.test_labels, self.test_preds, average='weighted')
        recall = recall_score(self.test_labels, self.test_preds, average='weighted')
        f1 = f1_score(self.test_labels, self.test_preds, average='weighted')
        cm = confusion_matrix(self.test_labels, self.test_preds)

        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> plt.Figure:
        """
        Plot a confusion matrix using Seaborn heatmap.
        
        Args:
            conf_matrix (np.ndarray): The confusion matrix to plot.
        
        Returns:
            plt.Figure: The matplotlib figure object containing the confusion matrix plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        return plt.gcf()

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            list: List of optimizers and learning rate schedulers.
        """
        # optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return [optimizer], [scheduler]


###### FinBERT and CryptoBERT ######

class TextDataset_fin(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_len: int) -> None:
        """
        Initializes the TextDataset_fin class.
        
        Args:
            dataframe (pd.DataFrame): The input dataset containing the text and labels.
            tokenizer (BertTokenizer): The tokenizer to be used for encoding the text.
            max_len (int): The maximum length for tokenized sequences.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """
        Retrieves a sample from the dataset and encodes the text.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Optional[dict]: A dictionary containing the input IDs, attention mask, and labels for the sample.
            Returns None if there is an error in the data.
        """
        try:
            row = self.dataframe.iloc[idx]
            text = row['combined_text']
            label = row['category_label']

            if pd.isnull(text) or pd.isnull(label):
                print(f"Missing value at index {idx}")
                return None

            assert label in [0, 1, 2, 3, 4], f"Invalid label: {label}"

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )

            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)

            assert input_ids.shape == (self.max_len,), f"Invalid input_ids shape: {input_ids.shape}"
            assert attention_mask.shape == (self.max_len,), f"Invalid attention_mask shape: {attention_mask.shape}"

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return None

class TextDataModule_fin(LightningDataModule):
    def __init__(
        self, 
        train_file: str, 
        validation_file: str, 
        test_file: str, 
        tokenizer_name: str, 
        batch_size: int = 32, 
        max_len: int = 128
    ) -> None:
        """
        Initializes the TextDataModule_fin class for loading and managing data.
        
        Args:
            train_file (str): Path to the training dataset.
            validation_file (str): Path to the validation dataset.
            test_file (str): Path to the test dataset.
            tokenizer_name (str): The name of the tokenizer to use.
            batch_size (int): The batch size for the dataloaders.
            max_len (int): Maximum token length for each sample.
        """
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage: Optional[str] = None) -> None:
        """Loads and prepares the datasets."""
        self.train_df = pd.read_csv(self.train_file)
        self.val_df = pd.read_csv(self.validation_file)
        self.test_df = pd.read_csv(self.test_file)

        self.train_dataset = TextDataset_fin(self.train_df, self.tokenizer, self.max_len)
        self.val_dataset = TextDataset_fin(self.val_df, self.tokenizer, self.max_len)
        self.test_dataset = TextDataset_fin(self.test_df, self.tokenizer, self.max_len)

    def collate_fn(self, batch: List[Optional[dict]]) -> torch.Tensor:
        """
        Collates the batch for the DataLoader, filtering out None values.

        Args:
            batch (List[Optional[dict]]): The input batch of data.

        Returns:
            torch.Tensor: The collated batch.
        """
        # Filter out None values and log batch size
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training dataset."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation dataset."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test dataset."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    

class FinBertClassifier(LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate: float) -> None:
        """
        Initializes the FinBertClassifier class for sequence classification.

        Args:
            model_name (str): The name of the pre-trained model.
            num_classes (int): Number of output classes.
            learning_rate (float): The learning rate for the optimizer.
        """
        super(FinBertClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.learning_rate = learning_rate

        self.test_preds = []
        self.test_labels = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): The input IDs for the model.
            attention_mask (torch.Tensor): The attention mask for the model.
            
        Returns:
            torch.Tensor: The logits for each class.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Defines the training step to compute the loss and accuracy for each batch.
        
        Args:
            batch (dict): The input batch of data.
            batch_idx (int): Index of the batch.
            
        Returns:
            torch.Tensor: The loss value for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        assert input_ids.dim() == 2, f"Invalid input_ids dimension: {input_ids.dim()}"
        assert attention_mask.dim() == 2, f"Invalid attention_mask dimension: {attention_mask.dim()}"
        assert labels.dim() == 1, f"Invalid labels dimension: {labels.dim()}"

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Defines the validation step to compute the loss and accuracy for each batch.

        Args:
            batch (dict): The input batch containing input_ids, attention_mask, and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        assert input_ids.dim() == 2, f"Invalid input_ids dimension: {input_ids.dim()}"
        assert attention_mask.dim() == 2, f"Invalid attention_mask dimension: {attention_mask.dim()}"
        assert labels.dim() == 1, f"Invalid labels dimension: {labels.dim()}"

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Defines the test step to compute the loss and accuracy for each batch.

        Args:
            batch (dict): The input batch containing input_ids, attention_mask, and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        assert input_ids.dim() == 2, f"Invalid input_ids dimension: {input_ids.dim()}"
        assert attention_mask.dim() == 2, f"Invalid attention_mask dimension: {attention_mask.dim()}"
        assert labels.dim() == 1, f"Invalid labels dimension: {labels.dim()}"

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Finalizes the test results, computes precision, recall, and F1 score, and plots the confusion matrix.

        This method is called after the test epoch ends and it calculates the overall performance metrics.
        """
        precision = precision_score(self.test_labels, self.test_preds, average='weighted')
        recall = recall_score(self.test_labels, self.test_preds, average='weighted')
        f1 = f1_score(self.test_labels, self.test_preds, average='weighted')
        cm = confusion_matrix(self.test_labels, self.test_preds)

        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> plt.Figure:
        """
        Plots the confusion matrix using Seaborn's heatmap.

        Args:
            conf_matrix (np.ndarray): The confusion matrix to be plotted.

        Returns:
            plt.Figure: The figure object containing the confusion matrix plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        return plt.gcf()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            torch.optim.Optimizer: The optimizer for training the model.
        """
        # return AdamW(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    



###### OpenAI Emnedding Model ######

# Define Dataset and DataModule for PyTorch Lightning
class EmbeddingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initializes the EmbeddingDataset class for loading and processing embedding data.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the embeddings and labels.
        """
        self.dataframe = dataframe

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of rows (samples) in the dataframe.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset based on its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the embedding and label for the sample.
        """
        row = self.dataframe.iloc[idx]
        embedding = eval(row['embedding'])  # Convert string to list
        embedding = np.array(embedding, dtype=np.float32)
        label = row['category_label']
        return {
            'embedding': torch.tensor(embedding, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = 32) -> None:
        """
        Initializes the EmbeddingDataModule class to handle loading and splitting data for training, validation, and testing.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset with embeddings and labels.
            batch_size (int, optional): Number of samples in each batch. Defaults to 32.
        """

        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            stage (Optional[str], optional): The stage (fit, validate, test) for which the datasets are set up. 
                                             Can be None if setting up for all stages.
        """
        train_df, temp_df = train_test_split(self.dataframe, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        self.train_dataset = EmbeddingDataset(train_df)
        self.val_dataset = EmbeddingDataset(val_df)
        self.test_dataset = EmbeddingDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader object for the training data.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader object for the validation data.
        """

        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader object for the test data.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class BertClassifier_emb(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float) -> None:
        """
        Initializes the BertClassifier_emb model for classification using embeddings.

        Args:
            num_classes (int): Number of output classes for classification.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(BertClassifier_emb, self).__init__()
        self.save_hyperparameters()
        self.classifier = torch.nn.Linear(1536, num_classes)  # the embedding size is 1536
        self.learning_rate = learning_rate

        self.test_preds = []
        self.test_labels = []

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier to obtain logits.

        Args:
            embedding (torch.Tensor): Embeddings tensor for the input data.

        Returns:
            torch.Tensor: Logits from the classifier.
        """
        logits = self.classifier(embedding)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Defines the training step to compute the loss and accuracy for each batch.

        Args:
            batch (dict): The input batch containing embeddings and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        embedding = batch['embedding']
        labels = batch['labels']
        logits = self(embedding)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Defines the validation step to compute the loss and accuracy for each batch.

        Args:
            batch (dict): The input batch containing embeddings and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        embedding = batch['embedding']
        labels = batch['labels']
        logits = self(embedding)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Defines the test step to compute the loss and accuracy for each batch.

        Args:
            batch (dict): The input batch containing embeddings and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        embedding = batch['embedding']
        labels = batch['labels']
        logits = self(embedding)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(torch.argmax(logits, dim=1), labels, task='multiclass', num_classes=self.hparams.num_classes)
        preds = torch.argmax(logits, dim=1)


        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Finalizes the test results, computes precision, recall, and F1 score, and plots the confusion matrix.

        This method is called after the test epoch ends and calculates the overall performance metrics.
        """
        precision = precision_score(self.test_labels, self.test_preds, average='weighted')
        recall = recall_score(self.test_labels, self.test_preds, average='weighted')
        f1 = f1_score(self.test_labels, self.test_preds, average='weighted')
        cm = confusion_matrix(self.test_labels, self.test_preds)

        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            List[torch.optim.Optimizer]: A list containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]

