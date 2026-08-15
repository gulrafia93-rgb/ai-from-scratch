# Day 11: Build My First Document Retrieval System

##  Goal
Build a retrieval engine on top of the Day 10 Pipeline, with a relevance
threshold that honestly reports when no document is actually relevant,
instead of always forcing a top-3 answer.

##  What I Built

Reused Day 10's `PreprocessingModule`, `VectorizerModule`, and `Pipeline`
classes, with one fix (see Bugs below), plus a new `retrieve()` function:

```python
def retrieve(query, pipeline, corpus, top_k=3, threshold=0.1):
    """
    Retrieves the most relevant documents for a query, with a relevance threshold.
    Parameters:
        query (str) - user's search query
        pipeline (Pipeline) - a fitted Pipeline instance
        corpus (list of str) - the knowledge base
        top_k (int) - number of results to return
        threshold (float) - minimum similarity score to trust a result
    Returns: list of (score, document) tuples, or a "no relevant document" message
    """
    results = pipeline.run(query, corpus, top_k=top_k)
    if results[0][0] < threshold:
        return "No relevant document found."
    return results
```

## 20-Document Knowledge Base
ML/NLP concepts spanning preprocessing, vectorization, model training,
and evaluation — see notebook for full list.

##  10-Query Test Results

| Query | Type | Top Score | Result |
|---|---|---|---|
| "What is tokenization?" | In-domain | 0.4028 |  Correct |
| "How does cosine similarity work?" | In-domain | 0.7420 | Correct |
| "What is supervised learning?" | In-domain | 0.4723 | Correct |
| "What is TF-IDF?" | In-domain | 0.4708 |  Correct |
| "What is lemmatization?" | In-domain | 0.3955 |  Correct |
| "What is a model?" | Ambiguous | 0.5005 |  Debatable (see analysis) |
| "How does learning work?" | Ambiguous | 0.2288 |  Weak (see analysis) |
| "What's the weather today?" | Out-of-domain | — | "No relevant document found" |
| "Best places to visit in Italy?" | Out-of-domain | — |  "No relevant document found" |
| "What is overfitting?" | In-domain | 0.3292 |  Correct |

##  Failure Analysis

**"What is a model?"** — TF-IDF selected the model-evaluation document
because the query shares the exact word "model" with it, while TF-IDF
cannot understand that other documents about training or neural networks
may be semantically relevant to the broader concept of a model.

**"How does learning work?"** — Although "learning" appears in several
documents, the query has little additional vocabulary overlap with any
single document, so TF-IDF distributes similarity across multiple
documents and produces relatively low scores.

##  Vocabulary Mismatch and Embeddings

TF-IDF mainly depends on vocabulary/exact word matching, so synonyms or
different word forms (e.g. "dog" vs "dogs," or "trained" vs "training")
can cause relevant documents to be missed entirely. Embeddings solve this
by representing words based on meaning and semantic relationships rather
than exact word matches. This shows that a real retrieval system
shouldn't just return top similarity results — it needs both a relevance
threshold and, ideally, semantic retrieval to reliably reject irrelevant
results and surface meaningful information.

##  Bug I Fixed
- `.isalpha()` was silently deleting hyphenated technical terms like
  "TF-IDF," causing a real query to return near-zero scores. Fixed by
  switching to `re.match(r'^[a-zA-Z-]+$', token)` to allow hyphens
  while still filtering numbers/symbols.

##  Status
Day 11 of #60DaysAIChallenge — Complete