# Tech FAQ

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI technique that combines information retrieval with text generation. Instead of relying solely on a model's training data, RAG fetches relevant documents from a knowledge base and uses them as context when generating answers. This makes responses more accurate, up-to-date, and grounded in specific source material.

## What is a Vector Embedding?
A vector embedding is a numerical representation of text (or images) as a list of floating-point numbers. Similar texts will have embeddings that are close together in vector space. This allows semantic search — finding documents that are *meaningfully* similar, not just keyword-matched.

## What is SQLite?
SQLite is a lightweight, file-based relational database. It stores all data in a single `.db` file and requires no separate server process. It is ideal for small to medium applications, prototypes, and embedded systems. Python's standard library includes built-in support for SQLite via the `sqlite3` module.

## How does cosine similarity work?
Cosine similarity measures the angle between two vectors. A score of 1.0 means the vectors are identical in direction (most similar), 0 means orthogonal (unrelated), and -1 means opposite. In NLP, it is used to rank document chunks by their similarity to a query embedding.

## What is sentence-transformers?
sentence-transformers is a Python library built on top of Hugging Face Transformers. It provides pre-trained models optimized for generating sentence and paragraph embeddings. Popular models include `all-MiniLM-L6-v2` (fast, small) and `snowflake-arctic-embed` (higher accuracy). These embeddings are used for semantic similarity, clustering, and retrieval tasks.
