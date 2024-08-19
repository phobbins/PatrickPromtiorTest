# PatrickPromtior

## Project Overview

It is a Retrieval-Augmented Generation (RAG) service designed to answer questions based on content scraped from specified websites. It uses the LlaMA2 model via the Ollama API, integrating with FastAPI for serving the responses.

## Installation

### Prerequisites

- Python 3.8 or higher
- FastAPI
- Uvicorn
- LangChain and its dependencies
- [Ollama](https://ollama.com) with the LlaMA2 model
- Optional: Poetry for dependency management

### Setup

1. Clone the Repository:

    ```bash
    git clone https://github.com/yourusername/patrickpromtior.git
    ```

2. Install the required packages:

    ```bash
    pip install fastapi uvicorn langchain langchain_core langchain_community langchain_huggingface
    ```

    If using Poetry:

    ```bash
    poetry install
    ```

3. Ensure Ollama and the LlaMA2 Model Server Are Running:

    Follow the instructions on the [Ollama website](https://ollama.com) to set up and run the Ollama server with the LlaMA2 model.


## Usage

1. Start the FastAPI server:

    ```bash
    poetry run langchain serve
    ```

    If you're not using Poetry, you can start the server directly with Uvicorn:

    ```bash
    uvicorn app.server:app --host 0.0.0.0 --port 8000
    ```

2. Access the server at:

    ```text
    http://localhost:8000/rag/playground/
    ```