
# PDF → ICD Chatbot (Streamlit)

This package contains a Streamlit app that uses an LLM + a PDF as the only knowledge source to extract ICD codes for symptom queries.
It also supports user feedback/corrections (stored locally) so the system can prefer corrected answers for future similar queries.

## Files
- `streamlit_app.py` - the main Streamlit application
- `requirements.txt` - Python dependencies
- `data/` - folder where indexed chunks, embeddings, and corrections are stored after you run the app

## How it works (short)
1. Upload your PDF via the sidebar "Index PDF". The app extracts text into chunks and computes embeddings.
2. Enter symptoms/questions in the main text area and click "Get ICD". The app finds relevant PDF chunks and asks the LLM to extract an ICD code **only from those chunks**.
3. If the answer is wrong, click ❌ and submit the correct ICD code. The correction is saved and used for similar future queries.

## Setup

1. Create a Python environment (Python 3.9+ recommended):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set your LLM/OpenAI API key in the environment:
```bash
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=sk-...
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

4. Open the Streamlit URL shown in the terminal (usually http://localhost:8501).

## Notes & safety
- The app **relies on the PDF containing explicit ICD codes** in text. If your PDF doesn't include ICD codes verbatim, the LLM will likely return `No ICD code found`.
- The app validates that the returned code appears in the retrieved snippets; if not, it rejects the LLM output.
- Corrections are stored locally in `data/corrections.json`. For production, use a proper DB and access controls.

## Customization
- To use a different LLM endpoint, set `LLM_API_URL` environment variable and ensure the payload format matches the API.
- You can change the embedding model by setting `EMBED_MODEL_NAME` env var before starting the app.

