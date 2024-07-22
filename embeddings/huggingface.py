"""HuggingFace embeddings model class."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import requests
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.pydantic_v1 import SecretStr

from dystematic.immensa_ip_search.embeddings.base import BaseEmbeddingsModel

OPEN_MODELS: Dict[str, str] = {
    "patent_sbert": "AI-Growth-Lab/PatentSBERTa",
    "uae_large": "WhereIsAI/UAE-Large-V1",
    "mini_v2": "sentence-transformers/all-MiniLM-L12-v2",
    "patent_sbert_v2": "AAUBS/PatentSBERTa_V2",
    "patent_embedding": "seongwoon/patent_embedding",
    "bge": "BAAI/bge-large-en-v1.5",
}


class HuggingFaceEmbeddingsModel(BaseEmbeddingsModel):
    """HuggingFace embeddings model."""

    name = "huggingface-embeddings"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-l6-v2",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create new HuggingFace embeddings model.

        Args:
            model_name (str, optional): model name.
                Defaults to "sentence-transformers/all-MiniLM-l6-v2".
            api_key (Optional[str], optional): HuggingFace API key. Defaults to None.
        """
        super().__init__(model_name)
        if api_key:
            self.model: Embeddings = HuggingFaceInferenceAPIEmbeddings(
                model_name=model_name,
                api_key=SecretStr(api_key),
                **kwargs,
            )
        else:
            self.model: Embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True},
                **kwargs,
            )


class CustomHuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, endpoint_url: str, api_key: str, model_name: str, **kwargs: Any):
        """Initialise the custom inference embeddings.

        Args:
            endpoint_url (str): URL of the inference endpoint
            api_key (str): API Key, HF token
            model_name (str): model name to run the inference with

        Returns:
            Embeddings
        """
        super().__init__(**kwargs)

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._check_inputs()

    def send_package(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create the client."""
        try:
            output = requests.post(
                self.endpoint_url, headers=self.headers, json=inputs
            ).json()
            return output[0]

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error in sending package: {e}")

    def _check_inputs(self) -> None:
        """Check the inputs for the embeddings"""
        if self.model_name not in OPEN_MODELS:
            raise ValueError(f"Model {self.model_name} not found")

    def embed_query(self, text: str) -> List[float]:
        """Embed the query"""
        result = self.send_package({"model_name": self.model_name, "inputs": [text]})
        return result["outputs"][0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed Documents"""
        result = self.send_package({"model_name": self.model_name, "inputs": texts})
        return result["outputs"]
