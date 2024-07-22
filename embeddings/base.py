"""Base embedding model class."""

from abc import ABC
from typing import ClassVar

from langchain_core.embeddings import Embeddings


class MissingEmbeddingModelNameError(Exception):
    """Thrown when the embedding model name is not configured."""


class BaseEmbeddingsModel(ABC):
    """Base embedding model class."""

    name: ClassVar[str]
    model_name: str
    model: Embeddings

    def __init_subclass__(cls) -> None:
        """Hook subclass initialization."""
        if cls.name is None:
            raise MissingEmbeddingModelNameError

    def __init__(self, model_name: str) -> None:
        """Create new embedding model."""
        self.model_name = model_name
