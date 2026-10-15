# Improvements and Known Limitations

## Potential improvements
- Add automated regression tests for prices, inclusions, source attribution, and safe fallback behavior.
- Improve retrieval with hybrid keyword and vector search, metadata filters, and systematic evaluation of chunk sizes and retrieval parameters.
- Add structured logging for unanswered questions and low-confidence retrievals to identify gaps in the knowledge base.
- Provide a controlled knowledge-base refresh process that re-ingests the approved source pages and rebuilds the vector collection.
- Add multilingual support, conversation history, and an optional consultation lead-capture workflow.

## Known limitations
The chatbot only has access to the selected PlacidWay pages and cannot reliably answer questions outside their content. Retrieval can miss relevant passages or return incomplete context. Generated responses depend on the availability and quota of the Gemini API. Package prices and program details may change, so the source data must be refreshed and verified. The chatbot is informational and must not diagnose conditions, prescribe treatment, or guarantee medical outcomes. Users should consult qualified medical professionals for individual decisions.
