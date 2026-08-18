# Day 13: Understanding Embeddings as Semantic Coordinates

##  Goal
Compare dense embeddings against sparse TF-IDF vectors on the same
sentences, prove embeddings generalize across vocabulary using real
paraphrase examples, and verify semantic clustering with K-means.

## Note on Tooling
Used `sentence-transformers` (free, local) instead of the OpenAI
Embeddings API, since no API key/billing was set up. Cost estimation
was calculated hypothetically using OpenAI's published per-token rate.

## 20-Sentence Corpus (4 topics x 5 sentences)
Sports, Technology, Cooking, Travel — with deliberate paraphrase pairs
planted within each topic to test vocabulary generalization.

##  TF-IDF vs Embedding: 5 Paraphrase Examples

| Pair | Sentences | TF-IDF | Embedding |
|---|---|---|---|
| (5,6) | AI automation vs ML efficiency | 0.0000 | 0.5874 |
| (15,16) | Tourists exploring vs visitors discovering | 0.1065 | 0.6656 |
| (0,1) | Football championship vs club trophy | 0.1095 | 0.6368 |
| (10,11) | Chef's pasta vs vegetables combined | 0.4780 | 0.8353 |
| (1,2) | Club trophy vs striker scoring | 0.0000 | 0.4452 |

## embed_and_recommend() Function

```python
def embed_and_recommend(query_sentence, corpus_sentences, top_k=3):
    """Finds the most semantically similar sentences using embeddings."""
    query_embedding = model.encode(query_sentence)
    corpus_embeddings = model.encode(corpus_sentences)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = np.argsort(scores.numpy())[::-1][:top_k]
    return [(scores[i].item(), corpus_sentences[i]) for i in top_results]
```

Tested with "The team celebrated their victory in the tournament" —
correctly matched football/sports sentences with zero shared vocabulary.

##  K-Means Clustering (4 clusters, no labels given)

19/20 sentences (95%) correctly grouped by real topic. Sports and
Travel clustered perfectly. One cooking sentence ("Baking bread
requires accurate measurements...") clustered with Technology instead
— explained by shared precision/process-oriented language, not a bug.

##  Performance
- Embedding generation time (20 sentences): ~0.6 seconds
- Estimated hypothetical OpenAI cost: ~$0.000006 (negligible at this scale)

##  Sparse vs Dense

TF-IDF represents a sentence based on which exact words are present —
sparse, mostly-zero vectors where each dimension is one vocabulary
word. Embeddings represent meaning and context as dense vectors where
every dimension is meaningful. TF-IDF asks "which words are here?";
embeddings ask "what does this mean?" This is why embeddings generalize
across vocabulary — they capture semantic relationships, not just word
presence, so paraphrases with completely different wording still land
close together in embedding space.

##  Key Takeaways
- Sparse vectors are mostly zeros with one slot per vocabulary word
- Dense embeddings compress meaning into fewer, fully-meaningful dimensions
- K-means clusters by embedding similarity, which can diverge from human topic labels in genuinely informative ways
- Embedding generation is fast and cheap at small scale, but cost/speed matter at production scale

## Status
Day 13 of #60DaysAIChallenge — Complete