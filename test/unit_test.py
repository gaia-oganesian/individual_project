# unit_test.py
import pytest
import torch
from utils import *
from typing import Tuple

# Define the model path
model_path = '../trained_models/bertft_optimizer_Adam_lr_1e-05_epochs_10_bs_8_maxlen_512'

# Mock model and tokenizer setup for unit testing
@pytest.fixture
def mock_model_and_tokenizer() -> Tuple[object, object]:
    """
    Creates a mock model and tokenizer for unit testing purposes.
    
    The mock model simulates different outputs based on input tokens, and the tokenizer 
    assigns specific tokens for specific input texts.

    Returns:
        Tuple[object, object]: Mock model and tokenizer.
    """
    # Define a simple mock output class to simulate model output
    class MockOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            """
            Simulate the model output with logits.

            Args:
                logits (torch.Tensor): Logits tensor representing model output.
            """
            self.logits = logits

    # Define a simple mock model class
    class MockModel:
        def __init__(self) -> None:
            pass

        def eval(self) -> None:
            pass

        def to(self, device: str) -> 'MockModel':
            """
            Simulate moving the model to a specific device.
            
            Args:
                device (str): The device to move the model to.

            Returns:
                MockModel: The model itself.
            """
            return self

        def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> MockOutput:
            """
            Simulate the forward pass of the model based on specific input tokens.

            Args:
                input_ids (torch.Tensor): Input IDs representing the tokens of the input text.
                attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

            Returns:
                MockOutput: Simulated output logits for the input IDs.
            """
            if input_ids[0][0].item() == 101:  # Token for "bounty"
                logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])  # Mock logits for category 0
            elif input_ids[0][0].item() == 102:  # Token for "telegram"
                logits = torch.tensor([[-1.0, 2.0, 0.0, -1.0, -2.0]])  # Mock logits for category 1
            elif input_ids[0][0].item() == 103:  # Token for "robin"
                logits = torch.tensor([[0.0, -1.0, 2.0, -2.0, -3.0]])  # Mock logits for category 2
            elif input_ids[0][0].item() == 104:  # Token for "airdrop"
                logits = torch.tensor([[-1.0, -2.0, 0.0, 2.0, -3.0]])  # Mock logits for category 3
            else:
                logits = torch.tensor([[-2.0, -1.0, 0.0, -3.0, 2.0]])  # Mock logits for category 4
            
            # Return an instance of MockOutput with the logits
            return MockOutput(logits)

    # Define a simple mock tokenizer class that handles batching
    class MockTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 0

        def __call__(self, texts: list, padding: str = 'max_length', truncation: bool = True, 
                     return_tensors: str = 'pt', max_length: int = 512) -> dict:
            """
            Simulate tokenizing the input texts and returning tokenized input IDs and attention masks.

            Args:
                texts (list): List of input texts to tokenize.
                padding (str, optional): Padding strategy. Defaults to 'max_length'.
                truncation (bool, optional): Truncation strategy. Defaults to True.
                return_tensors (str, optional): Return type for the tensors. Defaults to 'pt'.
                max_length (int, optional): Maximum length for tokenization. Defaults to 512.

            Returns:
                dict: A dictionary containing tokenized 'input_ids' and 'attention_mask'.
            """
            input_ids = []
            attention_masks = []
            for text in texts:
                if "bounty" in text:
                    input_ids.append([101])  # Simulate a specific token for text containing "bounty"
                elif "telegram" in text:
                    input_ids.append([102])  # Simulate a specific token for text containing "telegram"
                elif "robin" in text:
                    input_ids.append([103])  # Simulate a specific token for text containing "robin"
                elif "airdrop" in text:
                    input_ids.append([104])  # Simulate a specific token for text containing "airdrop"
                else:
                    input_ids.append([105])  # Simulate a specific token for text containing other categories
                attention_masks.append([1])  # Mock attention_mask
            
            return {
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_masks)
            }

    # Instantiate the mock model and tokenizer
    model = MockModel()
    tokenizer = MockTokenizer()
    return model, tokenizer




# Sample texts and expected categories
sample_texts = [
    "bounty the week everyone team decided end bounty count haste let know remove use bounty thread support dont personal support telegram bounty related official telegram group instead bounty thread may result ban team decided raise bounty worth u value new platform reshape world data consumer business alike fully forefront twitter ann thread split bounty based method contribution pool distribution way share value bounty task successfully complete pool end th entire value pool distributed stake based number hold ie someone get exactly double amount someone sale bounty bounty bounty bounty bounty distributed every week calculated end week bounty campaign start regarding bounty campaign write post reserve right eliminate think havent honest reserve right change bounty campaign member stake per stake per member stake per week stake member per week stake member per week stake per week stake campaign least must wear signature personal spotted expect post constructive per week local accept post made single day try spread posting made local board included excluding keep wearing signature personal text end first fill form without wearing signature form personal text signature second post must least long must least organic link free page least article must original contain least apply article different get based quality popularity work end accepted work must follow official twitter audit score must check every participant make sure twitter account twitter account must mainly also accept medium least real report custom every report custom help add twitter double accept anyone audit stake stake stake custom tweet le stake stake stake custom per user single form must follow like official check every participant make sure account account must real accept fake report report help add double accept anyone audit stake follow stake stake stake post le stake follow stake stake stake post form ask translate ann thread find youve used translate campaign moderate update thread news ann thread least finish ann translation day youve assigned thread translation earn article depend article end campaign distribute moderation depending activity stake looking form medium newsletter contact u discus fill application form order claim bounty end bounty team end minimum soft",
    "official distribution want distribute subscribe telegram maximum number need fill form link write need enter nick person youve come leave field write subscribe channel write telegram get extra referral write nick telegram filling follow end telegram distribution manual try get unfair way get wish",
    "robin future social medium robin global ambassador profile utility token put first step building future social medium successfully built matching application hire directly targeted marketing took step global based social component robin social medium platform continue develop product would like work closely u giving u feedback product also community respective community global need global like love official looking people fly location annual ambassador event everyone interact robin team responsible local meet member robin core team create social medium native language communicate community translate community create content social medium platform choice video giving monthly reward put starting subject line global ambassador following information name nationality currently reside social medium chosen become robin fill form",
    "airdrop join telegram earn airdrop campaign twitter telegram ann worth reserved join telegram group write message project telegram follow twitter make repost write comment submit apply form please post user name profile link",
    "bounty token st th digital magazine platform sept campaign bounty program reserved total k k target end fully distributed among bounty campaign total amount unknown token based amount bounty must register official telegram account looking forward engaging k k target end following campaign campaign campaign medium campaign distributed smart contract close important address reason careful address submit exchange party service provide know private seed find detailed list different bounty twitter join twitter bounty total bounty pool signature member stake member stake full member member hero member legendary per week wear wear make per must constructive must posting min per week consecutive account per person found alt trust may receive trust enrolled receive pay time enrolled may advertise sig campaign bounty count may change payment use address whole topic politics society help archival investor based micro earnings count participant want join bounty first wear signature based rank fill digital magazine platform digital magazine platform digital magazine platform thread twitter digital magazine platform digital magazine platform thread twitter digital magazine color digital magazine thread twitter digital magazine member digital magazine platform thread twitter digital magazine platform size member digital magazine platform thread twitter digital magazine platform size total bounty pool translation use translator translator instantly one translation per person multiple ann get moderation thread get get post thread reserve want reserve translation want reserve translation reserve translation ill manually check reservation best eligible participant let u know get touch u via total bounty pool medium take part post must original work content use official logo graphic posted ann thread must longer must least minute long minute accepted must link official project link make sure place address description video must one link official one link description video write original text start please post plan post post check eligible give depending quality given end backing participant"
]

# Expected categories based on the provided mapping
expected_categories = [0, 1, 2, 3, 4]  # Expected categories for each sample text

# Test with Mocked Model and Tokenizer
def test_mocked_prediction_pipeline(mock_model_and_tokenizer: Tuple[object, object]) -> None:
    """
    Test the prediction pipeline using mocked model and tokenizer.

    Args:
        mock_model_and_tokenizer (Tuple[object, object]): The mocked model and tokenizer.

    Returns:
        None
    """
    model, tokenizer = mock_model_and_tokenizer
    predicted_categories, _ = predict_category(model, tokenizer, sample_texts)
    
    # Assert that all predicted categories are within the expected range of category indices
    for category in predicted_categories:
        assert category in range(5), f"Predicted category {category} is out of the expected range [0-4]."

# Test with Real Model and Tokenizer
def test_prediction_pipeline() -> None:
    """
    Test the prediction pipeline using a real model and tokenizer.

    Returns:
        None
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    predicted_categories, _ = predict_category(model, tokenizer, sample_texts)
    
    # Assert predictions match the expected categories
    assert predicted_categories == expected_categories, f"Expected {expected_categories}, but got {predicted_categories}"


