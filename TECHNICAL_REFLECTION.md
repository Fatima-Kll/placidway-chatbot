# Technical Reflection: PlacidWay Website Knowledge Chatbot

## Potential Improvements with One Additional Week

With one additional week, I would focus on improving answer reliability, retrieval quality, and evaluation coverage. First, I would expand the test suite to cover different question types, including pricing comparisons, program inclusions, follow-up care, out-of-scope questions, and questions whose answers are not available in the indexed content. I would use these tests to identify retrieval errors and unsupported responses systematically.

Second, I would improve the retrieval pipeline by experimenting with chunk sizes, overlap, and retrieval parameters, and by evaluating whether reranking improves the relevance of retrieved passages. I would also improve source attribution so that responses clearly identify the pages supporting each answer. For questions requiring exact program details, I would continue prioritizing verified source content over general language-model knowledge.

Finally, I would improve error handling, add clearer feedback when the language-model API is unavailable, and evaluate response quality using metrics such as retrieval relevance, answer correctness, and groundedness.

## Current Limitations

The chatbot currently relies on a limited collection of selected PlacidWay pages, so it cannot reliably answer questions about information that has not been indexed. Its answer quality depends on the relevance and completeness of retrieved content, and language-model responses may still require further validation. API availability and quota restrictions can also affect response generation. As the application concerns medical-tourism information, it is designed to provide website information rather than personalized medical advice or guarantees of treatment outcomes. Further testing and broader source coverage would be necessary before using it in a production environment.
