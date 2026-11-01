from src.config import CHUNK_SIZE
from src.config import CHUNK_OVERLAP

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    "Découpe un texte en morceaux de caracteres avec overlap"
    chunks=[]
    step = size - overlap
    for start in range(0,len(text),step):
        chunk = text[start:start+size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def chunk_pages(pages):
    """Découpe chaque page et garde, pour chaque chunk, le fichier et la page d'origine."""
    chunks = []
    for p in pages : 
        for piece in chunk_text(p['text']):
            chunks.append({'source':p['source'], 'page': p['page'], 'text': piece})
    return chunks