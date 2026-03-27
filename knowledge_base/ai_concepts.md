# AI & Machine Learning Concepts

## What is a Large Language Model (LLM)?
A Large Language Model is a type of neural network trained on vast amounts of text data to understand and generate human language. LLMs like GPT-4, Claude, and Llama are trained using self-supervised learning on internet-scale corpora. They predict the next token in a sequence, which allows them to write code, answer questions, summarize documents, and much more.

## What is Fine-tuning?
Fine-tuning is the process of taking a pre-trained model and continuing to train it on a smaller, domain-specific dataset. This adapts the model to a new task without training from scratch, which saves enormous amounts of time and compute. Common fine-tuning approaches include full fine-tuning, LoRA (Low-Rank Adaptation), and QLoRA for quantized models.

## What is Prompt Engineering?
Prompt engineering is the practice of crafting effective inputs (prompts) for language models to achieve desired outputs. Techniques include zero-shot prompting, few-shot prompting (providing examples), chain-of-thought (asking the model to reason step by step), and role prompting (giving the model a persona). Good prompt engineering can dramatically improve output quality without any model training.

## What is Ollama?
Ollama is an open-source tool that allows you to run large language models locally on your own machine. It supports models like LLaMA 3, Mistral, Phi-3, Gemma, and many others. Ollama provides a simple CLI and a REST API, making it easy to integrate local LLMs into applications. It handles model downloading, quantization, and GPU acceleration automatically.

## What is Quantization?
Quantization reduces the precision of a model's weights (e.g., from 32-bit floats to 4-bit integers) to decrease memory usage and increase inference speed. A 7B parameter model that normally requires ~28GB in full precision can be run in under 5GB with 4-bit quantization. The quality tradeoff is usually minimal for general use cases, making quantized models practical for consumer hardware.
