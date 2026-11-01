from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

_model = None


def embed(texts):
    """Turn a list of texts into normalized vectors (the model is loaded once)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50)