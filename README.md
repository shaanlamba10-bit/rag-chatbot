# RAG Chatbot

A Streamlit document question-answering project. It retrieves relevant passages from PDF or TXT documents using TF-IDF and cosine similarity, then uses an OpenAI model to generate answers grounded in those passages. Source citations are included.

## Architecture

PDF/TXT → text extraction → overlapping chunks → TF-IDF retrieval → relevant passages → LLM generation → cited answer.

The project uses lexical retrieval, not neural embeddings or a persistent vector database. Without an API key, it runs in retrieval-only mode and clearly labels retrieved passages. A built-in sample document allows visitors to try retrieval immediately.

## Local setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter an OpenAI API key in the sidebar to enable generation. API charges may apply. Alternatively, set `OPENAI_API_KEY` in Streamlit secrets. Never commit credentials.

## Deploy to Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ and choose Create app.
2. Select this repository, main branch and `app.py` as the entrypoint.
3. In Advanced settings, optionally configure a secret using `OPENAI_API_KEY = "your-key"`. A public deployment using a shared key may incur costs; using the sidebar key avoids sharing your own credential with visitors.
4. Deploy and open the assigned `.streamlit.app` URL. Copy that actual URL into your submission only after testing it.

## Try the chatbot

- Keep the sample document enabled and ask: “What does the retriever do?”
- Upload a TXT file containing your own notes and ask a relevant question.
- With an API key, verify the generated answer cites retrieved passages.
- Ask an unrelated question and check that the app does not invent an answer.

## Tests

Run `python -m unittest discover -v` to test document chunking and retrieval. Full Streamlit UI and live model generation should be tested separately after deployment.

## Limitations and privacy

Scanned PDFs require OCR, which is not included. Retrieval depends on lexical overlap. If generation is enabled, relevant document passages are sent to OpenAI. Do not upload sensitive documents without permission. No keys or uploaded documents are included in this repository.
