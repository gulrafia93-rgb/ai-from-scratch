# Day 14: Build a Semantic Search Engine (FAISS)

## Goal
Build a production-style semantic search engine using FAISS to store
and search embedding vectors, and compare it directly against a
keyword search baseline on the same 50-document corpus.

## Note on Tooling
Used `sentence-transformers` instead of the OpenAI Embeddings API
(billing not available). Cost/latency analysis reasons about the
production pattern generally, based on published API pricing and
measured local performance.

## Corpus
50 short documents covering ML fundamentals, evaluation metrics, NLP
preprocessing, embeddings, FAISS/retrieval, and neural networks.

##  What I Built

```python
embeddings = model.encode(documents).astype('float32')  # (50, 384)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
# index.ntotal == 50

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [(dist, documents[idx]) for dist, idx in zip(distances[0], indices[0])]

def keyword_search(query, corpus, top_k=3):
    query_words = set(query.lower().split())
    scores = [len(query_words & set(doc.lower().split())) for doc in corpus]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], corpus[i]) for i in top_indices]
```

##  10-Query Comparison Summary

| Winner | Queries |
|---|---|
| Semantic search | "How do computers understand language?", "Prevent overfitting?", "Purpose of embeddings in NLP?" |
| Keyword search | "Precision vs recall?", "Semantic similarity between text?", "FAISS fast similarity search?" |
| Both correct | 4 remaining queries |

## Failure Analysis

**Semantic wins** happen on vocabulary mismatch — e.g. "prevent overfitting"
correctly matched the overfitting document, while keyword search matched
generic ML documents sharing only common words.

**Keyword wins** happen on exact technical terminology — e.g. "precision
vs recall" was matched precisely by keyword search (exact terms present),
while semantic search's top result was the conceptually-related but
less exact "accuracy" document.

##  Written Trade-off Analysis

Semantic search excels when the query and document express the same
idea in different words — it understands meaning, not just vocabulary.
Keyword search excels when exact terminology matters (technical terms,
IDs, error messages). At production scale, embeddings are generated
once (indexing cost) and reused, while each query only requires a
single new embedding (query cost) — this is why FAISS-based search
scales far better than manual comparison loops. A real production
system should use hybrid search: combining keyword and semantic
results (optionally with a reranker) captures the strengths of both,
rather than committing to one method exclusively.

## Key Takeaways
- FAISS L2 distance: lower = more similar (opposite of cosine similarity)
- Semantic search generalizes across vocabulary; keyword search is precise on exact terms
- Indexing cost (one-time) vs query cost (per-search) is a critical production distinction
- Hybrid search (keyword + semantic + reranking) is the real-world answer, not choosing one

##  Status
Day 14 of #60DaysAIChallenge — Complete