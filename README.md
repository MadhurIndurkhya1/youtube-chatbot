# 🎥 YouTube RAG Chat

> **A conversational Retrieval-Augmented Generation (RAG) system that lets users chat with YouTube videos using hybrid retrieval, cross-encoder reranking, metadata-aware routing, multilingual responses, and persistent conversation architecture.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-purple.svg)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**YouTube RAG Chat** is a conversational AI application that allows users to paste a YouTube video URL and interact with the video's content through natural-language conversation.

Instead of sending the entire transcript to an LLM, the system uses a **Retrieval-Augmented Generation pipeline** to retrieve only the most relevant sections of the transcript before generating an answer.

The system combines:

* 🎬 YouTube transcript ingestion
* 🧠 Semantic vector retrieval
* 🔎 BM25 keyword retrieval
* 🔀 Hybrid retrieval
* 🎯 Cross-encoder reranking
* 🧭 Metadata-aware question routing
* 🌍 Automatic language detection
* 💬 Conversational follow-up questions
* 🗂️ Multi-chat conversation history
* 🤗 Hugging Face / local model support
* 🖥️ Gradio chat interface

The project is designed as a practical implementation of a **production-oriented RAG architecture**.

---

# 🚀 Key Features

## 1. YouTube Video Ingestion

Users simply paste a YouTube URL.

The application extracts:

* Video ID
* Title
* Channel
* Description
* Duration
* View count
* Upload information
* Transcript

Example:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

---

## 2. Intelligent Metadata Routing

Not every question should go through vector retrieval.

For example:

```text
User:
Which channel uploaded this video?

        ↓

Metadata Router

        ↓

YouTube Metadata

        ↓

Answer
```

Whereas:

```text
User:
What is the main concept explained in the video?

        ↓

RAG Pipeline

        ↓

Transcript Retrieval

        ↓

Reranking

        ↓

LLM

        ↓

Answer
```

This prevents unnecessary retrieval when the required information already exists in video metadata.

---

# 🧠 RAG Architecture

```text
                         YouTube URL
                              │
                              ▼
                     ┌─────────────────┐
                     │   yt-dlp        │
                     │ Video Metadata  │
                     └────────┬────────┘
                              │
                              │
                              ▼
                     ┌─────────────────┐
                     │   Transcript    │
                     │     Loader      │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │     Chunking    │
                     │ Recursive Split │
                     └────────┬────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          ┌─────────────┐           ┌─────────────┐
          │    FAISS    │           │    BM25     │
          │   Semantic  │           │   Keyword   │
          │  Retrieval  │           │  Retrieval  │
          └──────┬──────┘           └──────┬──────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                     ┌─────────────────┐
                     │ Hybrid Retrieval│
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Cross Encoder   │
                     │    Reranker     │
                     └────────┬────────┘
                              │
                              ▼
                        Top Relevant
                           Chunks
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Metadata + Context +   │
                 │ Conversation History  │
                 └────────────┬───────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │      LLM        │
                     │ Hugging Face /  │
                     │     Ollama      │
                     └────────┬────────┘
                              │
                              ▼
                     Conversational
                         Response
```

---

# 🔍 Retrieval Pipeline

The retrieval system uses multiple retrieval techniques instead of relying on a single vector search.

## Semantic Retrieval

FAISS retrieves chunks based on semantic similarity.

```python
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)
```

This helps when the user uses different words with the same meaning.

Example:

```text
Transcript:
"Gradient descent minimizes the loss function."

Question:
"How does the model reduce its error?"
```

Semantic retrieval can recognize the conceptual relationship.

---

## Keyword Retrieval

BM25 handles lexical matching.

```python
bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 10
```

This is especially useful for:

* Names
* Technical terms
* Numbers
* Algorithms
* Exact phrases
* Product names

---

# 🔀 Hybrid Retrieval

Both retrieval methods are combined using an ensemble retriever.

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever
    ],
    weights=[
        0.4,
        0.6
    ]
)
```

The system therefore benefits from both:

```text
BM25
 ↓
Lexical relevance

+

FAISS
 ↓
Semantic relevance
```

---

# 🎯 Cross-Encoder Reranking

Retrieval produces candidate documents, but the first ranking isn't always optimal.

The project therefore performs a second-stage reranking step.

```python
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

The reranker evaluates:

```text
(query, document)
```

pairs and assigns relevance scores.

The pipeline becomes:

```text
Retrieve 10–20 candidates
          ↓
      Reranker
          ↓
   Top 3–5 chunks
          ↓
         LLM
```

This reduces irrelevant context being passed to the generation model.

---

# 🧭 Metadata-Aware Routing

The system separates **video metadata questions** from **content questions**.

### Metadata question

```text
"Which channel uploaded this video?"
```

→ Metadata

### Content question

```text
"What is the main concept explained?"
```

→ Hybrid RAG

### Follow-up question

```text
"Why is that important?"
```

→ Conversation history + rewritten query + RAG

This routing architecture reduces unnecessary retrieval and makes the system more reliable.

---

# 💬 Conversational Memory

The application supports multi-turn conversations.

Example:

```text
User:
What is RAG?

Assistant:
RAG combines retrieval with generation...

User:
Why do we need embeddings?

Assistant:
Embeddings allow the system to represent...

User:
Explain that with an example.

Assistant:
For example...
```

The system rewrites follow-up questions into standalone questions before retrieval.

```text
Conversation History
        +
Latest Question
        ↓
Question Rewriting
        ↓
Standalone Question
        ↓
Hybrid Retrieval
```

---

# 🌍 Multilingual Support

The system detects the language of the transcript and user question.

For example:

```text
English Video
      +
Hindi Question
      ↓
Hindi Answer
```

or:

```text
Hindi Video
      +
Hindi Question
      ↓
Hindi Answer
```

The answer language is passed explicitly to the generation prompt.

---

# 🖥️ User Interface

The application uses Gradio to provide a conversational interface.

The UI is designed around a ChatGPT-style workflow:

```text
┌──────────────────────────────────────────────────────┐
│ YouTube RAG Chat                                     │
├──────────────┬───────────────────────────────────────┤
│              │                                       │
│ + New Chat   │ Video Title                           │
│              │ Channel: Example Channel              │
│ Previous     │                                       │
│ Conversations│ ┌───────────────────────────────────┐ │
│              │ │ User: What is RAG?                │ │
│ Video A      │ │                                   │ │
│ Video B      │ │ AI: RAG combines...               │ │
│ Video C      │ │                                   │ │
│              │ └───────────────────────────────────┘ │
│              │                                       │
│              │ Ask something about the video...      │
└──────────────┴───────────────────────────────────────┘
```

Each video conversation maintains its own context.

---

# 🛠️ Technology Stack

| Component      | Technology             |
| -------------- | ---------------------- |
| Language       | Python                 |
| RAG Framework  | LangChain              |
| LLM            | Hugging Face / Ollama  |
| Embeddings     | Sentence Transformers  |
| Vector Search  | FAISS                  |
| Keyword Search | BM25                   |
| Reranking      | Cross Encoder          |
| Transcript     | YouTube Transcript API |
| Metadata       | yt-dlp                 |
| UI             | Gradio                 |
| Environment    | python-dotenv          |

---

# 📁 Project Structure

```text
youtube-rag-chat/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
└── data/
    └── generated-at-runtime/
```

For a larger production version, the application can be separated into:

```text
youtube-rag-chat/
│
├── app/
│   ├── main.py
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── metadata/
│   └── memory/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-chat.git

cd youtube-rag-chat
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

Add:

```env
HF_TOKEN=your_huggingface_token
```

Never commit `.env` to GitHub.

Use:

```text
.env.example
```

for public repositories:

```env
HF_TOKEN=your_token_here
```

---

# ▶️ Run the Application

```bash
python app.py
```

Gradio will provide a local URL such as:

```text
http://127.0.0.1:7860
```

Open it in your browser.

---

# 🧪 Example Questions

After loading a video, users can ask:

### Video metadata

```text
Which channel uploaded this video?
```

```text
What is the title of the video?
```

```text
How long is this video?
```

### Content

```text
What is the main idea of the video?
```

```text
Explain the concept of embeddings.
```

```text
What example does the speaker give?
```

### Follow-up questions

```text
Why is that important?
```

```text
Explain the second point in more detail.
```

```text
Can you give me a practical example?
```

### Summarization

```text
Summarize the entire video.
```

```text
Give me the five most important takeaways.
```

---

# 🧠 Why RAG Instead of Sending the Entire Transcript?

A long YouTube transcript can contain thousands of words.

Sending the entire transcript to an LLM for every question is inefficient.

Instead:

```text
Large Transcript
       ↓
    Chunking
       ↓
   Indexing
       ↓
 User Question
       ↓
 Retrieval
       ↓
 Relevant Chunks
       ↓
      LLM
       ↓
    Answer
```

This reduces irrelevant context and makes the application more scalable.

---

# 📊 Retrieval Strategy

The project uses a two-stage retrieval architecture.

### Stage 1 — Recall

```text
BM25 + FAISS
```

Goal:

> Find a broad set of potentially relevant chunks.

### Stage 2 — Precision

```text
Cross Encoder
```

Goal:

> Identify the most relevant chunks among the candidates.

This follows the general principle:

```text
Retrieve broadly
        ↓
Rerank precisely
        ↓
Generate carefully
```

---

# 🔒 Security Considerations

The project follows several basic security practices:

* API keys stored in environment variables
* `.env` excluded from Git
* No secrets embedded in source code
* Transcript treated as untrusted content
* Prompt-injection instructions inside transcripts are not treated as system instructions
* User-facing UI does not expose internal retrieval data

---

# 🚧 Current Limitations

This project is currently designed as a strong RAG prototype.

Current limitations include:

* Conversation history is stored in application memory.
* FAISS indexes are generated per loaded video.
* The application is currently designed for a single application process.
* Large reranker models can require significant RAM.
* YouTube transcript availability varies between videos.
* `yt-dlp` metadata extraction can be affected by YouTube changes.
* Production deployment requires persistent storage and user/session management.

---

# 🔮 Production Roadmap

The next version can evolve into a production architecture:

```text
                    Frontend
                       │
                       ▼
                    FastAPI
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Metadata     RAG Engine    Chat Memory
          │            │            │
        yt-dlp    ┌─────┴─────┐   PostgreSQL
                   │           │
                 BM25      pgvector
                   │           │
                   └─────┬─────┘
                         │
                     Reranker
                         │
                         ▼
                        LLM

---

# 📈 Evaluation Strategy

A production version should not rely only on whether the chatbot "feels good."

Important RAG metrics include:

### Retrieval

* Recall@K
* Precision@K
* MRR
* NDCG

### Generation

* Faithfulness
* Answer relevance
* Context relevance
* Groundedness

### System

* Latency
* Token usage
* Cost per query
* Throughput
* Error rate

The goal is to evaluate the complete pipeline rather than only the LLM.

---

# 🏗️ Engineering Concepts Demonstrated

This project demonstrates practical knowledge of:

* Retrieval-Augmented Generation
* Semantic search
* Dense embeddings
* Vector databases
* Lexical search
* BM25
* Hybrid retrieval
* Cross-encoder reranking
* Query rewriting
* Metadata routing
* Prompt engineering
* Multilingual NLP
* Conversational memory
* LangChain LCEL
* Hugging Face models
* FAISS
* Gradio
* Environment management
* RAG system architecture

---

# 💼 Resume Description

### YouTube RAG Chat — Conversational Video Intelligence System

**Tech:** Python, LangChain, FAISS, BM25, Sentence Transformers, Cross-Encoder, Hugging Face, Gradio, yt-dlp

> Built a conversational YouTube RAG system that combines semantic FAISS retrieval and BM25 keyword search with cross-encoder reranking to improve contextual retrieval. Implemented metadata-aware routing, multilingual question handling, query rewriting, and multi-turn conversation management through a ChatGPT-style interface.

### Short Resume Version

> Developed a multilingual YouTube RAG chatbot using hybrid BM25 + FAISS retrieval, cross-encoder reranking, metadata routing, query rewriting, and conversational memory.

---

# 📚 Learning Outcomes

This project was designed to demonstrate the complete RAG lifecycle:

```text
Data Ingestion
      ↓
Document Processing
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Search
      ↓
Keyword Search
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Query Rewriting
      ↓
Context Construction
      ↓
LLM Generation
      ↓
Conversation Memory
      ↓
Evaluation
      ↓
Production Deployment
```

---

# 👨‍💻 Author

**Madhur Indurkhya**

B.Tech — Computer Science & Engineering
Artificial Intelligence & Machine Learning

Interested in:

* Machine Learning
* Deep Learning
* Generative AI
* Retrieval-Augmented Generation
* Agentic AI
* MLOps

---

# 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
