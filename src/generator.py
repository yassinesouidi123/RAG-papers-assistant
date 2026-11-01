import ollama
from src.config import LLM_MODEL


def build_prompt(question, results):
    """Assemble les extraits numérotés et la question dans un seul prompt."""
    context = "\n\n".join(
        f"[{i}] ({r['source']}, p.{r['page']})\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )
    return f"""Answer the question using ONLY the extracts below.
Cite the extracts you use as [1], [2], etc.
If the answer is not in the extracts, say "I don't know based on these documents."

Extracts:
{context}

Question: {question}"""


def generate(question, results):
    """Demande au LLM local de répondre à partir des extraits déjà trouvés."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
    )
    return response["message"]["content"]