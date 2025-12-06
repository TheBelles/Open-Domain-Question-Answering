# Retrieval-Augmented Generation (RAG) with Custom GPT Model

This repository implements and evaluates a Retrieval-Augmented Generation (RAG) system built around a custom GPT-style transformer for open-domain question answering. The project integrates transformer modeling, document retrieval, prompting strategies, and downstream fine-tuning.

---

## Model Architecture

We build a GPT-small decoder-only transformer with the following components:

- Rotary Positional Embeddings (RoPE)
- RMSNorm normalization layers
- Grouped Query Attention (GQA)
- Pretraining on a large text corpus
- Fine-tuning on:
  - Summarization (CNN/DailyMail)
  - Classification (20 Newsgroups)

---

## RAG Pipeline Overview

The system combines retrieval and generation through:

### BM25 Retriever
Used to fetch top-k relevant documents from a text corpus for each query.

### Language Models
- Custom GPT-small model  
- Llama-3.2-1B-Instruct baseline  

Retrieved passages are dynamically inserted into the model’s context window to improve grounding and factual accuracy.

---

## Techniques Explored

### Prompting Strategies
- Few-shot prompting  
- Chain-of-thought reasoning (CoT)

### Context Compression
Techniques explored to increase relevance per token:
- Instruction-aware compression  
- Extractive summarization of retrieved text

---

## Key Findings

Our results show:

- RAG improves access to up-to-date information  
- Model size and prompting strategy significantly affect performance  
- Context compression improves efficiency in limited context windows  
- Experiments highlight challenges and opportunities in retrieval-augmented systems

---

## Summary

This project includes:

- A custom GPT architecture  
- A full RAG pipeline implementation  
- Comparative evaluation with an instruction-tuned LLM  
- Exploration of prompting, retrieval, and compression techniques  

Together, these components demonstrate both the strengths and limitations of modern RAG systems for open-domain question answering.

---
