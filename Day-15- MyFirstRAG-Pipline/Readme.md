Day 15 — Build My First RAG Pipeline

Overview

This project demonstrates how to build a basic Retrieval-Augmented Generation (RAG) pipeline by connecting a FAISS semantic search engine with an LLM.

The goal was to test whether providing retrieved knowledge as context can reduce LLM hallucinations compared to direct generation.

Instead of asking the LLM to answer only from its training knowledge, the RAG pipeline retrieves relevant information first and provides it as context to the LLM.

 Objectives

Connect the Day 14 FAISS semantic search engine with an LLM

Build generate_with_rag()

Build generate_without_rag()

Compare RAG vs. non-RAG responses

Test RAG using custom fictional knowledge

Identify RAG failure cases

Understand how retrieval affects hallucination

Create a complete RAG architecture

 What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.

Without RAG

User Query

LLM

Answer

With RAG

User Query
    
Query Embedding
    
FAISS Vector Search
    
Top 3 Relevant Chunks
    
Context + Query
    
LLM
    

RAG Pipeline Architecture

                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           ↓
                  ┌─────────────────┐
                  │ Query Embedding │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │  FAISS Search   │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ Top 3 Chunks    │
                  └────────┬────────┘
                           ↓
              ┌────────────────────────┐
              │ Context + User Query   │
              └───────────┬────────────┘
                          ↓
                  ┌───────────────┐
                  │      LLM      │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │ Final Answer  │
                  └───────────────┘

 Core Functions

generate_with_rag(query)

Retrieves the top 3 relevant chunks from FAISS, formats them as context, and sends the context together with the user's question to the LLM.

def generate_with_rag(query):
    results = semantic_search(query, top_k=3)

    context = "\n\n".join(
        [document for distance, document in results]
    )

    prompt = f"""
    Answer the question using ONLY the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    If the answer is not available in the context,
    say that you don't have enough information.
    """

    response = llm.invoke(prompt)
    return response

generate_without_rag(query)

Sends only the question directly to the LLM without retrieving external context.

def generate_without_rag(query):
    response = llm.invoke(query)
    return response

 Experiment

I created a fictional knowledge base containing information that the LLM could not reliably know from its training data.

Observation

The RAG pipeline performed better when answering questions about fictional/custom information that was not available in the LLM's original knowledge.

 RAG Failure Analysis

RAG reduces hallucination, but it does not completely eliminate errors.

Failure Case 1 — Wrong Retrieval

Query: What is the warranty period of X1?

RAG Answer: 1 year

Correct Answer: 2 years

Root Cause: FAISS retrieved a chunk about the return policy, which mentioned 1 year, instead of the correct warranty information.

Query
  ↓
Wrong Chunk Retrieved
  ↓
LLM Uses Wrong Context
  ↓
Wrong Answer

Lesson: Poor retrieval quality can directly cause incorrect generation.

Failure Case 2 — Generation Drift

Query: Does X1 support ECG monitoring?

RAG Answer: Yes, it supports ECG.

Correct Answer: No, X1 does not support ECG.

Root Cause: The retrieved chunk contained information about another model, X2, which supported ECG. The LLM mixed information from the retrieved context and generated an incorrect answer.

Lesson: Even when retrieval works, the LLM can sometimes drift away from the exact context.

 Key Findings

Without RAG

❌ LLM relies on existing knowledge

❌ Custom information may be unavailable

❌ Higher risk of hallucination

❌ Can generate confident but incorrect answers

With RAG

✅ External knowledge can be injected into the prompt

✅ Answers become more grounded

✅ Custom knowledge can be used

✅ Retrieved evidence can be inspected

✅ Hallucination can be reduced

However:

RAG does not automatically guarantee correct answers.

The quality of a RAG system depends on both retrieval and generation.

Retrieval Quality
        +
Generation Quality
        ↓
RAG Answer Quality

 What I Learned

This project helped me understand:

Semantic search

Vector embeddings

FAISS vector search

Context retrieval

Prompt engineering

Retrieval-Augmented Generation

LLM grounding

Hallucination reduction

Retrieval failure

Generation drift

RAG evaluation

One of the most important lessons was:

A RAG system is only as reliable as the information it retrieves and how faithfully the LLM uses that information.

 Future Improvements

🔹 Hybrid Search

🔹 Better document chunking

🔹 Metadata filtering

🔹 Reranking retrieved documents

🔹 Retrieval evaluation metrics

🔹 Citation generation

🔹 Conversation memory

🔹 Streaming responses

🔹 Better hallucination evaluation

🔹 Production-ready vector database

 Conclusion

Building this RAG pipeline demonstrated an important principle in AI engineering:

Don't just make an LLM generate. Give it the right information first.

RAG combines:

Search
  +
Retrieval
  +
Context
  +
Generation
  =
Grounded AI Responses

The experiment showed that providing relevant context can significantly improve answers to questions about custom knowledge.

At the same time, the failure cases demonstrated that retrieval accuracy and generation discipline are both critical for building reliable RAG systems.