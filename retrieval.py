"""Document chunking and lexical retrieval, independent of Streamlit."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text, source, size=220, stride=180):
    words = text.split()
    return [(' '.join(words[i:i + size]), source) for i in range(0, len(words), stride)]


def retrieve(question, passages, limit=4):
    if not passages or not question.strip():
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
    try:
        vectors = vectorizer.fit_transform([text for text, _ in passages])
    except ValueError:
        return []
    scores = cosine_similarity(vectorizer.transform([question]), vectors)[0]
    return [passages[int(i)] for i in scores.argsort()[::-1][:limit] if scores[i] > 0]
