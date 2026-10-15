# PlacidWay Website Knowledge Chatbot

## Overview
A retrieval-augmented generation (RAG) chatbot that answers questions using a curated knowledge base of seven PlacidWay medical-tourism pages related to ITC Immunity Therapy Center. Responses are grounded in retrieved page content, with source links shown where available.

The chatbot is an informational demonstration, not a medical diagnosis or treatment recommendation service.

## Architecture
1. **Source data:** `data/pages.json` stores the seven scraped pages; `data/chunks.json` stores the processed text chunks and their source metadata.
2. **Embeddings:** ChromaDB's `SentenceTransformerEmbeddingFunction` uses `all-MiniLM-L6-v2`.
3. **Vector store:** ChromaDB stores chunks and retrieves relevant content by semantic similarity.
4. **Answer generation:** Python response logic handles specific questions, with Google Gemini used for supported general responses when appropriate.
5. **User interface:** Streamlit provides the chat interface and displays source references.

On first launch, the application creates the ChromaDB collection from `data/chunks.json` if the collection is empty.

## Requirements
- Python 3.10 or newer
- A Google Gemini API key for AI-generated responses

## Local setup
1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The app builds its local ChromaDB collection from the included chunks on first launch.

## Knowledge-base refresh
The source ingestion and chunking scripts are provided in the repository. Review the source URLs and ingestion rules before refreshing the knowledge base. After updating the chunks, rebuild the collection so that retrieval reflects the new content.

## Evaluation
Local checks cover package pricing, package inclusions, package comparisons, unsupported medical claims, personalized medical recommendations, unavailable contact details, out-of-scope questions, and incorrect price assumptions. See `TEST_RESULTS.md` for the test log.

## Limitations
- Answers are limited to the content available in the selected source pages.
- Missing or ambiguous information should trigger a cautious fallback rather than an invented answer.
- Medical decisions must be made with a qualified healthcare professional.
- Gemini API availability and quotas may affect generated responses.
- The local ChromaDB directory is generated at runtime and is not committed to the repository.

## Security
Never commit `.env`, API keys, credentials, or local virtual-environment files. For cloud deployment, configure `GEMINI_API_KEY` using the hosting provider's secrets settings.
