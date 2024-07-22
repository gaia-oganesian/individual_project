"""Init file for embeddings."""

from .huggingface import HuggingFaceEmbeddingsModel
from .openai import OpenAIEmbeddingsModel

__all__ = ["OpenAIEmbeddingsModel", "HuggingFaceEmbeddingsModel"]
