markdown

# Day 5: Build a Mini Search Engine

## Goal

Build a working semantic search engine using sentence embeddings, and
compare its results against traditional keyword search on the same
query — the same core pattern behind RAG, semantic recommendations,
and modern search products.

## What I Built

### Corpus (20 sentences, 2 clusters: fast food vs healthy food)

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    "Fast food is quick and easy to eat.",
    "Fruits are healthy and delicious.",
    "Burgers are a popular fast food.",
    "Vegetables are good for our health.",
    "Pizza is a type of fast food.",
    "Drinking water keeps us healthy.",
    "Fast food often contains a lot of salt.",
    "Eggs are a good source of protein.",
    "Eating too much fast food is not good for health.",
    "Milk is good for our bones.",
    "Fast food is usually tasty and convenient.",
    "Healthy food gives us energy.",
    "Many people enjoy eating fast food with friends.",
    "Eating fruits every day is a healthy habit.",
    "We should not eat fast food every day.",
    "A balanced diet keeps our body strong.",
    "French fries are a common fast food.",
    "Whole grains are good for our body.",
    "Fast food can contain a lot of sugar and fat.",
    "Healthy food helps us stay active."
]

embeddings = model.encode(sentences)  # (20, 384) - computed once
```

### Semantic Search Function

```python
def semantic_search(query, top_k=3):
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, embeddings)
    top_results = torch.topk(scores, k=top_k)
    print("Top", top_k, "results for query:", query)
    for idx in top_results.indices[0]:
        print(sentences[idx])
```

### Keyword Search Function (for comparison)

```python
def keyword_search(query):
    query_words = query.lower().split()
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for word in query_words:
            if word in sentence_lower:
                print(sentence)
                break
```

##  Example: Query = "junk food"

**Semantic search (top 3):**
Fast food can contain a lot of sugar and fat. (0.5401)
Eating too much fast food is not good for health. (0.4837)
Healthy food gives us energy. (0.4706)

**Keyword search:** matched 13+ sentences, including several irrelevant
healthy-food sentences (matched only because they contained the word "food").

##  Reflection

The main difference between semantic search and keyword search is that
semantic search embeds the query and compares it against stored sentence
embeddings, so results are based on meaning rather than exact words.
Keyword search is based purely on matching search words, and often gives
wrong or irrelevant results because those words frequently appear in
unrelated sentences too — like how "food" matched both fast food and
healthy food sentences, even though the query was specifically about
"junk" food.

One weakness of semantic search: it can rank topically related sentences
highly even when their meaning is opposite. "Healthy food gives us energy"
appeared in the top 3 results for "junk food" — likely because both share
the broader topic of food and health, even though one is about junk food
and the other about healthy eating.

##  Key Takeaways

- Corpus embeddings are computed once and reused; the query is embedded fresh each search
- `torch.topk()` returns both top scores and their original indices
- Semantic search finds meaning-based matches keyword search misses
- Semantic search isn't perfect — it can conflate "related topic" with "same meaning"

## Status

Day 5 of #60DaysAIChallenge — Complete 