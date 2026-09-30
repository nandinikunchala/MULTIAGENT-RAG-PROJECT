# Multi-Agent RAG Research Assistant

A fully local **Multi-Agent Retrieval-Augmented Generation (RAG) Research Assistant** built with **Python, LangGraph, FAISS, Sentence Transformers, FastAPI, and a local language model**.

The system uses three specialized agents:

**Retriever Agent → Summarizer Agent → Critic Agent**

The agents work together through a LangGraph workflow. The Retriever finds relevant information from the document knowledge base, the Summarizer generates a grounded answer, and the Critic evaluates the answer. If the answer requires improvement, the workflow sends it back for revision within a fixed revision budget.

The project demonstrates how **RAG, multi-agent orchestration, vector search, answer evaluation, revision loops, testing, and API deployment** can be combined into a complete AI application.

---

## Table of Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Solution](#solution)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [How the System Works](#how-the-system-works)
* [Agents](#agents)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Local RAG Pipeline](#local-rag-pipeline)
* [Configuration](#configuration)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Using the API](#using-the-api)
* [Example Request](#example-request)
* [Example Response](#example-response)
* [Evaluation](#evaluation)
* [Testing](#testing)
* [Infinite Loop Prevention](#infinite-loop-prevention)
* [Why LangGraph](#why-langgraph)
* [Why Multi-Agent Architecture](#why-multi-agent-architecture)
* [Design Principles](#design-principles)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Learning Outcomes](#learning-outcomes)
* [Conclusion](#conclusion)

---

# Overview

The Multi-Agent RAG Research Assistant is a document-grounded question-answering system.

Instead of generating answers only from a language model's internal knowledge, the system first retrieves relevant information from a document knowledge base and then generates an answer based on that retrieved information.

A critic agent evaluates the generated answer and can request a revision when the answer is not sufficiently supported.

The complete workflow is orchestrated using **LangGraph**.

The project is designed to run in **local mode without requiring paid API services**.

---

# Problem Statement

A standard language model may produce answers that:

* contain unsupported information
* miss important information from a document
* provide irrelevant responses
* generate plausible but incorrect statements
* have no explicit mechanism for checking the generated answer

A simple RAG pipeline improves grounding by retrieving relevant documents before generation, but it still does not explicitly evaluate whether the generated answer is sufficiently supported.

This project addresses the problem by combining:

* Retrieval-Augmented Generation
* Vector similarity search
* Specialized agents
* Answer evaluation
* Critic-driven revision
* Conditional workflow routing
* Automated testing
* Retrieval evaluation

---

# Solution

The system separates the research process into specialized responsibilities.

```text
User Question
      |
      v
Retriever Agent
      |
      v
Relevant Documents
      |
      v
Summarizer Agent
      |
      v
Generated Answer
      |
      v
Critic Agent
      |
      +------------------+
      |                  |
      v                  v
    PASS               REVISE
      |                  |
      v                  v
Final Answer      Summarizer Agent
                         |
                         v
                  Revised Answer
```

This architecture allows each agent to perform one clearly defined task.

---

# Key Features

## Local RAG

The project supports a fully local RAG pipeline using:

* Sentence Transformers
* FAISS
* Local document processing
* Local language-model inference

No paid API key is required for local mode.

## Multi-Agent Workflow

The system contains three specialized agents:

1. Retriever Agent
2. Summarizer Agent
3. Critic Agent

Each agent has a specific responsibility.

## Vector Similarity Search

Document chunks are converted into embeddings and stored in a FAISS vector index.

When a user asks a question, the question is embedded and compared against stored document vectors.

The highest-scoring documents are retrieved.

## Critic-Driven Revision

The Critic Agent checks whether the generated answer is relevant and sufficiently supported.

If the answer is not acceptable, the workflow sends it back to the Summarizer Agent.

## Revision Budget

The system has a maximum revision limit.

This guarantees that the workflow terminates even when the Critic repeatedly requests revisions.

## FastAPI Interface

The project exposes the RAG workflow through a REST API.

The API provides:

* health checking
* question answering
* structured JSON responses

## Automated Testing

The project includes unit and workflow tests using pytest.

The tests cover:

* RAG question answering
* Retriever Agent
* Summarizer Agent
* Critic Agent
* LangGraph workflow
* Revision handling
* Infinite-loop prevention

## Retrieval Evaluation

A separate evaluation pipeline measures retrieval quality using a predefined test set.

---

# System Architecture

The system consists of several layers.

```text
                 +------------------+
                 |    User / API    |
                 +---------+--------+
                           |
                           v
                 +------------------+
                 |  LangGraph Flow  |
                 +---------+--------+
                           |
              +------------+------------+
              |                         |
              v                         v
       Retriever Agent          Workflow State
              |
              v
        FAISS Vector Store
              |
              v
       Retrieved Documents
              |
              v
       Summarizer Agent
              |
              v
          Draft Answer
              |
              v
         Critic Agent
              |
        +-----+-----+
        |           |
       PASS       REVISE
        |           |
        v           v
      Final     Summarizer
```

---

# How the System Works

## Step 1 — Document Ingestion

The source document is loaded and its text is extracted.

The project uses the document:

```text
paper.pdf
```

The extracted text is divided into smaller chunks.

Chunking makes the document easier to search efficiently.

---

## Step 2 — Embedding Generation

Each document chunk is converted into a numerical vector using a Sentence Transformer model.

The project uses:

```text
all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the text.

---

## Step 3 — Vector Storage

The generated embeddings are stored in a FAISS vector index.

FAISS enables efficient similarity search over the document embeddings.

---

## Step 4 — Question Retrieval

When a user submits a question:

```text
What is the role of the retriever agent?
```

the question is converted into an embedding.

The vector store compares the question embedding against document embeddings.

The most relevant chunks are returned.

---

## Step 5 — Summarization

The retrieved documents are passed to the Summarizer Agent.

The Summarizer generates an answer based on the retrieved context.

The goal is to keep the answer grounded in the retrieved information.

---

## Step 6 — Criticism

The Critic Agent evaluates the generated answer.

It checks whether the answer:

* addresses the question
* contains relevant information
* is supported by retrieved information

The Critic returns either:

```text
PASS
```

or:

```text
REVISE
```

---

## Step 7 — Revision

If the Critic returns `REVISE`, the answer is sent back to the Summarizer Agent.

The Summarizer generates an improved answer.

This process continues only within the configured revision budget.

---

## Step 8 — Final Response

Once the Critic returns `PASS`, the final answer is returned through the API.

The response contains:

* original question
* final answer
* critic decision
* number of revisions
* critic feedback

---

# Agents

## Retriever Agent

The Retriever Agent is responsible for document retrieval.

### Responsibilities

* Accept the user's question
* Generate the query embedding
* Search the vector store
* Retrieve the most relevant document chunks

Conceptually:

```text
Question
   ↓
Query Embedding
   ↓
FAISS Similarity Search
   ↓
Top-K Documents
```

---

## Summarizer Agent

The Summarizer Agent generates the answer.

### Responsibilities

* Receive the user's question
* Receive retrieved documents
* Build relevant context
* Generate a grounded response

The Summarizer does not perform document retrieval itself.

---

## Critic Agent

The Critic Agent evaluates the generated answer.

### Responsibilities

* Check relevance
* Check grounding
* Detect insufficient answers
* Approve the answer
* Request revision when required

The Critic creates the feedback signal used by the LangGraph workflow.

---

# Technology Stack

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Core programming language          |
| LangGraph             | Multi-agent workflow orchestration |
| FAISS                 | Vector similarity search           |
| Sentence Transformers | Text embeddings                    |
| FastAPI               | REST API                           |
| Pydantic              | API request/response validation    |
| PyPDF                 | PDF text extraction                |
| Pytest                | Automated testing                  |
| Uvicorn               | FastAPI server                     |
| Local LLM             | Local answer generation            |

---

# Project Structure

```text
MULTIRAG PROJECT/
│
├── app/
│   ├── agents.py
│   ├── config.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── llm.py
│   └── vectorstore.py
│
├── data/
│   └── ...
│
├── eval/
│   ├── evaluate.py
│   └── testset.py
│
├── tests/
│   └── test_graph.py
│
├── .venv/
│
├── api.py
├── app.py
├── test_app.py
├── test_pipeline.py
├── test_retrieval.py
├── requirements.txt
├── README.md
├── paper.pdf
└── extracted_text.txt
```

---

# Local RAG Pipeline

The local retrieval pipeline follows:

```text
PDF
 |
 v
Text Extraction
 |
 v
Text Chunking
 |
 v
Sentence Transformer
 |
 v
Embeddings
 |
 v
FAISS Index
 |
 v
Similarity Search
 |
 v
Relevant Documents
```

The vector store uses semantic similarity rather than simple keyword matching.

This allows questions and relevant document passages to be matched based on their meaning.

---

# Configuration

Configuration is managed through:

```text
app/config.py
```

Important configuration values include:

```python
MODE = "local"
TOP_K = 3
MAX_REVISIONS = 2
```

## MODE

Controls the execution mode.

The local mode is designed to run without paid external services.

## TOP_K

Controls the number of documents retrieved for a question.

Example:

```text
TOP_K = 3
```

means the system retrieves the three highest-ranked results.

## MAX_REVISIONS

Controls the maximum number of times an answer can be revised.

Example:

```text
MAX_REVISIONS = 2
```

This prevents an endless Critic → Summarizer cycle.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MULTIRAG-PROJECT
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate the Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

The project can be executed locally.

Make sure the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run the application components as required by the project configuration.

For the API server:

```bash
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# Using the API

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface provides access to:

```text
GET  /
GET  /health
POST /ask
```

---

# Health Check

Endpoint:

```text
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

---

# Ask a Question

Endpoint:

```text
POST /ask
```

Request:

```json
{
  "question": "What is the role of the retriever agent?"
}
```

Example response:

```json
{
  "question": "What is the role of the retriever agent?",
  "answer": "The Retriever Agent performs the retrieval part of the RAG system. It searches the vector store and retrieves the most relevant documents for the user's question.",
  "decision": "PASS",
  "revisions": 0,
  "feedback": "Answer correctly describes the Retriever agent."
}
```

---

# Example Questions

The system can answer questions such as:

```text
What is RAG retrieval?

What are the three agents in this Multi-Agent RAG system?

What is the role of the retriever agent?

What is the role of the summarizer agent?

What is the role of the critic agent?

How do the three agents work together?

Why is LangGraph used in this project?

How does the system prevent infinite loops?

What is the purpose of the Multi-Agent RAG system?

How is retrieval evaluated?
```

---

# Evaluation

Retrieval quality is evaluated using a predefined test set.

The evaluation process:

1. Loads the document store.
2. Loads evaluation questions.
3. Defines the expected source document.
4. Retrieves the top-K documents.
5. Checks whether the expected document appears in the retrieved results.
6. Calculates retrieval accuracy.

The evaluation command is:

```bash
python -m eval.evaluate
```

Example output:

```text
========== RETRIEVAL EVALUATION ==========

Test 1: PASS
Test 2: PASS
Test 3: PASS
Test 4: PASS
Test 5: PASS
Test 6: PASS

========== RESULTS ==========

Correct: 6/6
Retrieval accuracy: 100.00%
```

The evaluation demonstrates that the expected source document was retrieved for all queries in the current evaluation set.

---

# Testing

The project uses **pytest** for automated testing.

Run the complete test suite with:

```bash
pytest -v
```

The tests cover:

```text
RAG question answering
Retriever Agent
Summarizer Agent
Critic Agent
LangGraph workflow
Revision budget
Infinite-loop prevention
```

A successful test run reports all tests as passing.

---

# Infinite Loop Prevention

A major design requirement of the system is preventing an endless revision cycle.

Without a limit, the workflow could theoretically behave like:

```text
Summarizer
    ↓
Critic
    ↓
REVISE
    ↓
Summarizer
    ↓
Critic
    ↓
REVISE
    ↓
...
```

The project solves this using a revision budget.

For example:

```text
MAX_REVISIONS = 2
```

The graph tracks the number of revisions.

Once the maximum number of revisions is reached, the workflow terminates.

This guarantees that the workflow cannot continue indefinitely.

---

# Why LangGraph?

A simple sequential chain would look like:

```text
Retriever → Summarizer → Critic
```

However, this project requires conditional routing.

The Critic must be able to decide:

```text
PASS
```

or:

```text
REVISE
```

The workflow therefore becomes:

```text
Retriever
    ↓
Summarizer
    ↓
Critic
   / \
PASS  REVISE
 |      |
 v      v
END  Summarizer
```

LangGraph is suitable for this architecture because it provides:

* shared workflow state
* graph-based execution
* conditional routing
* agent handoffs
* controlled loops

---

# Why Multi-Agent Architecture?

The project intentionally separates responsibilities instead of using one large agent.

Each agent has a clear role:

```text
Retriever
    |
    | finds information
    v
Summarizer
    |
    | generates answer
    v
Critic
    |
    | evaluates answer
    v
PASS / REVISE
```

This separation provides several benefits:

* easier debugging
* clearer responsibilities
* easier testing
* controlled workflow
* explicit evaluation
* easier future extension

The architecture also makes it possible to improve or replace individual components independently.

---

# Design Principles

## Separation of Responsibilities

Each agent performs one primary task.

## Grounded Generation

Answers are generated using retrieved document information.

## Explicit Evaluation

The Critic Agent provides an explicit evaluation signal.

## Controlled Iteration

Revision is allowed only within a predefined budget.

## Testability

Individual components and the complete workflow are tested independently.

## Local-First Design

The project can operate locally without depending on paid external APIs.

## Provider-Agnostic Architecture

The workflow separates orchestration, retrieval, embeddings, and language-model interfaces so that components can be changed independently.

---

# Limitations

The current implementation has several limitations.

### Limited Evaluation Dataset

The current evaluation set is relatively small and is intended primarily to demonstrate the retrieval evaluation workflow.

### Single Primary Document

The current project is demonstrated using the provided research document.

A larger document collection would provide a stronger production-style evaluation.

### Local Model Performance

Local language models may be slower or less capable than large hosted models depending on available hardware.

### Retrieval Quality

The quality of generated answers depends strongly on the quality of retrieved document chunks.

Poor retrieval can result in incomplete answers.

### Simple Critic

The Critic Agent currently uses rule-based checks and document grounding signals rather than a sophisticated independent evaluation model.

---

# Future Improvements

Possible future improvements include:

* support for multiple PDF documents
* persistent FAISS indexes
* improved chunking strategies
* metadata filtering
* source citations in final answers
* larger evaluation datasets
* Recall@K and Precision@K metrics
* more sophisticated critic evaluation
* improved local language models
* streaming responses
* authentication
* Docker deployment
* frontend interface
* production logging
* monitoring and observability
* configurable embedding models
* configurable vector databases

---

# Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector embeddings
* Semantic search
* FAISS
* Sentence Transformers
* Multi-agent systems
* LangGraph
* Agent state management
* Conditional routing
* Critic-driven revision
* FastAPI
* REST APIs
* Automated testing
* Retrieval evaluation
* Python virtual environments
* Local AI model execution

---

# Conclusion

The Multi-Agent RAG Research Assistant demonstrates a complete AI engineering workflow from document ingestion to retrieval, generation, evaluation, revision, testing, and API serving.

The core architecture is:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS
   ↓
Retriever Agent
   ↓
Summarizer Agent
   ↓
Critic Agent
   ↓
PASS / REVISE
   ↓
Final Answer
```

The project combines **RAG retrieval, multi-agent orchestration, controlled revision, evaluation, and API deployment** into a single modular system.

It provides a practical foundation for building more advanced document-grounded AI research assistants.

---

## License

This project is intended for educational and research purposes.

Add an appropriate open-source license such as MIT if you plan to publish the repository publicly.