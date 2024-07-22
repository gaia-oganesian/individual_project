"""Open AI embeddings model class."""

from typing import Any
from typing import Dict
from typing import Final
from typing import Literal
from typing import Optional

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from dystematic.immensa_ip_search.embeddings.base import BaseEmbeddingsModel
from research.config import OPENAI_API_KEY

AZURE_MODEL_DEPLOYMENT: Final[Dict[str, str]] = {
    "text-embedding-ada-002": "oaid-ada-switzerlandnorth",
}


class OpenAIEmbeddingsModel(BaseEmbeddingsModel):
    """OpenAI embeddings model."""

    name = "openai-embeddings"

    def __init__(
        self,
        openai_api_key: str = OPENAI_API_KEY,
        model_name: str = "text-embedding-ada-002",
        openai_api_type: Literal["openai", "azure"] = "openai",
        deployment: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init the CustomOpenAIEmbeddings class.

        Raises:
            ValueError: if the azure model deployment is not configured.
                Only when applicable.
        """
        super().__init__(model_name)
        if openai_api_type == "azure":
            if model_name not in AZURE_MODEL_DEPLOYMENT:
                raise ValueError(
                    f"No azure deployment configured for model: {model_name}"
                )

            deployment = AZURE_MODEL_DEPLOYMENT[model_name]

        self.model: Embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key,
            openai_api_type=openai_api_type,
            deployment=deployment,
            **kwargs,
        )
