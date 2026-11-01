import numpy as np
from src.config import TOP_K
from src.embedder import embed

def search(question, chunks , embeddings, k=TOP_K):
    """Renvoie les k chunks les plus proches de la question, avec leur score."""
    q = embed([question])[0]               # vecteur de la question (normalisé, 384 nombres)
    scores = embeddings @ q                 # similarité cosinus avec TOUS les chunks d'un coup
    top = np.argsort(scores)[::-1][:k]      # indices des k meilleurs scores, du plus haut au plus bas
    return [{**chunks[i], "score": float(scores[i])} for i in top]
