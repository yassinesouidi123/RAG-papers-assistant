import os 
from pypdf import PdfReader

def load_pdfs(folder):
    """Lit tous les PDF d'un dossier et renvoie une liste de pages."""
    pages = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(folder, filename))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():          # ignore les pages vides
                pages.append({"source": filename, "page": i, "text": text})
    return pages

