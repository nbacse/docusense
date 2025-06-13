# DocuSense – AI File Summarizer & QA

Upload PDF, DOCX, or TXT documents to:
- Extract text content
- Summarize long texts
- Ask questions about the content using LLM

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Setup

1. Upload to GitHub.
2. Deploy on https://streamlit.io/cloud.
3. Add this secret:

    HUGGINGFACEHUB_API_TOKEN = "your_token_here"

## Dependencies

| Name                       | Purpose                        |
| -------------------------- | ------------------------------ |
| `transformers`             | Summarization pipeline         |
| `langchain + faiss`        | QA over documents              |
| `PyMuPDF`, `docx`          | Text extraction                |

## License
MIT
