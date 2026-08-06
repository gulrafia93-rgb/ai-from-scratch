# Day 4: Vectors - Teaching Meaning to Machines

##  Goal
Learn how embeddings turn sentences into vectors that capture meaning,
and use cosine similarity to measure how closely related two sentences
are — the foundation of semantic search, recommendations, and RAG.

##  What I Built

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = [
    "I love Pizza",
    "Pizza is my favorite food",
    "Cats are sleeping",
    "The stock market crashed",
    "I enjoy eating pizza"
]

embeddings = model.encode(sentence)

score = util.cos_sim(embeddings[0], embeddings[1])
print("Score:", score)
```

##  Similarity Results

| Sentence pair | Score | Relationship |
|---|---|---|
| "I love Pizza" vs "Pizza is my favorite food" | 0.8851 | Highly similar |
| "I love Pizza" vs "I enjoy eating pizza" | 0.8496 | Highly similar |
| "I love Pizza" vs "The stock market crashed" | 0.0580 | Unrelated |
| "I love Pizza" vs "Cats are sleeping" | 0.0034 | Unrelated |
| "The stock market crashed" vs "Cats are sleeping" | -0.0193 | Unrelated, slightly opposite |

##  Reflection

I expected both pizza sentences to get the exact same similarity score
since they mean almost the same thing, but they scored differently
(0.88 vs 0.85). This showed me that the model weighs full sentence
phrasing, not just the topic word. I also noticed unrelated sentences
scored slightly negative rather than exactly zero, showing the model
can detect when two things are pointing in somewhat opposite directions
in meaning.

##  Key Takeaways
- Embeddings represent meaning as vectors, not just word counts
- Sentences with similar meaning land close together in vector space,
  even with completely different words
- Cosine similarity: closer to 1 = similar, closer to 0 = unrelated,
  negative = somewhat opposite
- Even sentences with near-identical meaning can score differently due
  to phrasing differences

## Bug I Fixed
- Typo'd model name (`MiniML` instead of `MiniLM`) caused a 401
  Unauthorized error — fixed by correcting the model name

## Status
Day 4 of #60DaysAIChallenge — Complete