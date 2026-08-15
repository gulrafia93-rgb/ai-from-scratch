# Day 12: Chunking Strategies for Long Document Processing

##  Goal
Compare manual fixed-size chunking against LangChain's recursive
character splitting across 12 configurations, and determine which
chunk size/overlap combination produces the most useful retrieval
results — not just the highest similarity score.

##  What I Built

### Manual fixed-size chunking (from scratch)
```python
def fixed_size_chunk(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks
```

### LangChain recursive splitting
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
recursive_chunks = splitter.split_text(document)
```

## 12-Configuration Experiment (chunk count)

| Chunk Size | Overlap 0 | Overlap 50 | Overlap 100 |
|---|---|---|---|
| 100 | 360 | 568 | **2667**  |
| 300 | 137 | 146 | 156 |
| 500 | 91 | 91 | 94 |
| 1000 | 43 | 43 | 43 |

##  Retrieval Comparison (same query, all 12 configs)

| Chunk Size | Best Score | Chunk Quality |
|---|---|---|
| 100 | **0.6941** (highest) | Cut off mid-sentence, incomplete answer |
| 300 | 0.4783–0.4832 |  Partial context |
| 500 | 0.4094–0.4147 |  Most complete, usable context |
| 1000 | 0.4694–0.4718 | Complete but less focused |

**Key finding:** Higher similarity score does not mean better retrieval —
size=100 scored highest but returned an incomplete, cut-off answer,
while size=500 (lower score) returned a genuinely complete, useful chunk.

##  Boundary Failure Examples

| Position | End of chunk | Start of next chunk | Issue |
|---|---|---|---|
| 300 | "...originally" | " described by t..." | Phrase split between related words |
| 500 | "...gies can be emp" | "loyed to genera..." | Word "employed" split mid-word |
| 800 | "...th rate, unifor" | "mity, and elect..." | Word "uniformity" split mid-word |

##  Written Recommendation

Higher similarity score does not always mean the most useful result —
smaller chunks can score higher due to focused content but often lack
complete context. A moderate chunk size (500) with modest overlap (50)
balances relevance, context, and completeness. Setting chunk_overlap
equal to chunk_size causes runaway redundancy (2667 chunks from a
100/100 configuration) without added value. Overlap's real purpose is
preventing information loss at boundaries, not boosting similarity
scores — proven by the boundary examples above, where words and
phrases were literally cut in half without overlap protection.

##  Key Takeaways
- Similarity score alone is a misleading metric for retrieval quality
- Chunk overlap should be a small fraction of chunk size, not equal to it
- Recursive splitting reduces (but doesn't eliminate) hard boundary cuts
- A good chunk size balances focus (small) against context (large)

##  Status
Day 12 of #60DaysAIChallenge — Complete