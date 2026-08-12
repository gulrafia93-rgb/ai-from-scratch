
# Day 10: Build Your First End-to-End NLP Pipeline

## Goal
Refactor individual NLP components (preprocessing, vectorization,
similarity) into modular, reusable classes connected through a single
Pipeline, with proper error handling for edge cases.

## Architecture

Query + Corpus
↓
PreprocessingModule
↓
VectorizerModule
↓
Pipeline (orchestrates both)
↓
Similarity Scoring
↓
Ranked Results (Top-K)

## What I Built

![End-To-End-NLP_pipeline](Day_10_My_First_End_to_End_NLP_Pipeline.ipynb)

## Testing: 5 Queries on 15-Document Corpus

All 5 queries correctly returned their most relevant document as the top
match (scores ranged 0.43–0.72 for top results). Full output in notebook.

## Edge Case Handling

| Input | Result |
|---|---|
| `""` (empty string) | `ValueError`: "Query has no valid words after cleaning..." |
| `"a"` (single character, stopword) | `ValueError`: same message |
| `"12345 !@#$%"` (symbols/numbers only) | `ValueError`: same message |

All 3 edge cases correctly raise a clear, catchable error instead of
silently returning meaningless zero-similarity results.

## Design Decision Note

Modular code is easier to debug and extend because each class has a
single, well-defined responsibility. Today, bugs appeared in
`Pipeline.run()` (line ordering and raw-vs-cleaned text issues), but I
never had to touch `PreprocessingModule` or `VectorizerModule` at all —
the bug was isolated to exactly one place. If everything had been in
one giant script, tracing the actual source of the bug would have been
much harder, since preprocessing, vectorization, and ranking logic would
all be tangled together.

##  Bugs I Fixed
- Method incorrectly named `preprocess_text` instead of the required `transform`
- Called `transform()` as a standalone function instead of on a class instance
- `VectorizerModule.transform()` used `.fit_transform()` instead of `.transform()`, rebuilding vocabulary incorrectly
- `Pipeline.run()` fit/transformed on raw text instead of cleaned text
- Placed the empty-query check before `cleaned_query` was even created (ordering bug)

## Status
Day 10 of #60DaysAIChallenge — Complete