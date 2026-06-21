Hello ! This is the set - up that i used to get data, and experiment for my research paper investigating the effect of changing the context size on RAG in Small
Language modeels (2.8B parameter model was used - I used Phi-2 by Microsoft )

Here is the file structure : 

rag-small-context
|
-main.py 
-requirements.txt 
-results.csv
-results_no_rag.csv 
-test.py 
-test_chunking.py 
-test_model.py 
-test_retreival.py 
-testing_dependencies.py 
---- Data 
     \ 
      -load_data.py 
      -__innit__.py 

 ---- models 
     \ 
      -load_model.py

---- Prompts
     \ 
      -context_controller.py
      -templates.py 
---- Results
     \ 
      -phi2_rag_results.csv
---- Evaluation
     \ 
      -metrics.py
      -evaluate_results.py 
      -__innit__.py 
---- Experiments
     \ 
      -run_experiment.py 
      -__innit__.py 
---- Retrieval 
     \ 
      -bm25_retreiver.py
      chunking.py
      -__innit__.py 
# Investigating the Effect of Context Window Size on Retrieval-Augmented Generation in Small Language Models

## Abstract

Large Language Models (LLMs) have demonstrated strong performance across a variety of natural language processing tasks. However, smaller language models with limited parameter counts often struggle with factual question answering due to restricted parametric knowledge and reduced reasoning capacity.

This project investigates the effect of Retrieval-Augmented Generation (RAG) and varying context window sizes on question-answering performance in small language models.

Experiments were conducted using the Microsoft Phi-2 language model (2.7B parameters) on the TriviaQA dataset. BM25 retrieval was used to provide external context, and context windows ranging from 128 to 512 tokens were evaluated using Exact Match (EM) as the primary metric.

Results demonstrate that RAG significantly improves factual question-answering performance compared to the non-RAG baseline. However, increasing context window size beyond smaller token limits produced minimal additional improvement, suggesting limitations in context utilization for small language models.

---

## Problem

Small language models often have reduced factual knowledge and reasoning ability compared to larger models.

Retrieval-Augmented Generation (RAG) improves performance by providing external information during inference, but the relationship between retrieved context size and model performance is not fully understood.

This project investigates:

* How much RAG improves factual accuracy in small language models
* How context window size affects retrieval-based generation
* Whether larger retrieved contexts always improve performance

---

## Method Overview

### Model

The experiments use:

* Model: Microsoft Phi-2
* Parameters: 2.7B
* Architecture: Transformer-based language model

### Dataset

Evaluation was performed on the TriviaQA question-answering dataset.

A subset of the dataset was used to allow experiments under limited computational resources.

### Retrieval System

BM25 retrieval was used to identify relevant context passages.

Configuration:

* Chunk size: 80 tokens
* Chunk overlap: 20 tokens

Retrieved passages were ranked by relevance and inserted into prompts.

### Experimental Setup

Two approaches were evaluated:

### No-RAG Baseline

The model answers questions using only its internal parametric knowledge.

### RAG

Retrieved context is provided to the model before generation.

Pipeline:

```
Question
   ↓
BM25 Retrieval
   ↓
Chunk Selection
   ↓
Prompt Construction
   ↓
Phi-2 Generation
   ↓
Exact Match Evaluation
```

---

## Results

| Context Window | No-RAG EM | RAG EM |
| -------------- | --------- | ------ |
| 128            | 0.2105    | 0.3684 |
| 160            | 0.2105    | 0.3684 |
| 192            | 0.2105    | 0.3684 |
| 224            | 0.2105    | 0.3684 |
| 256            | 0.2105    | 0.3684 |
| 288            | 0.2105    | 0.3684 |
| 320            | 0.2105    | 0.3684 |
| 352            | 0.2105    | 0.3684 |
| 384            | 0.2105    | 0.3684 |
| 416            | 0.2105    | 0.3684 |
| 448            | 0.2105    | 0.3684 |
| 480            | 0.2105    | 0.3684 |
| 512            | 0.2105    | 0.3684 |

## Findings

* RAG improved Exact Match accuracy from **0.2105 → 0.3684**
* Retrieved context substantially improved factual answering
* Increasing context window size did not produce additional gains
* Small language models may have difficulty effectively utilizing larger retrieved contexts

---

## Installation

Clone the repository:

```bash
git clone <repository>
cd <repository>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the experiment:

```bash
python main.py
```

The implementation performs:

1. Load TriviaQA samples
2. Retrieve relevant chunks using BM25
3. Construct RAG prompts
4. Generate answers using Phi-2
5. Evaluate using Exact Match

Experiment settings such as context window size can be modified in the configuration files.

---

## Citation

If you use this work, please cite:

```bibtex
@article{rajesh2026ragcontext,
  title={Investigating the Effect of Context Window Size on Retrieval-Augmented Generation in Small Language Models},
  author={Rajesh, Dev},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20780725},
  url={https://doi.org/10.5281/zenodo.20780725}
}
```

## Paper

DOI:

https://doi.org/10.5281/zenodo.20780725


