# Uni Assistant

Uni Assistant is my backend prototype for university admission guidance.

The idea of the project is to build an AI assistant that can help future students understand university programs, requirements, and study structure based on official sources.

At the current stage, this is still an early version focused on one university program and one PDF document.

## Project goal

This project currently has two main goals:

1. to be a portfolio project for IT job applications  
2. to become a practical prototype of an AI assistant for university admission guidance

Long term idea:
an assistant for CIS / post-Soviet applicants who want to apply to European universities without agencies.

## Current scope

This version currently works with:

- FastAPI
- one PDF source
- RAG
- FAISS vector search
- OpenAI model
- contextual planning for broader questions

This is not a production-ready system.
It is a working prototype.

## Current architecture

Current pipeline:

PDF -> chunks -> embeddings -> FAISS -> retrieval -> GPT answer

Main file:
- `app.py`

Main components:
- `PyPDFLoader` for PDF loading
- `RecursiveCharacterTextSplitter` for chunking
- `OpenAIEmbeddings` for embeddings
- `FAISS` for vector search
- `ChatOpenAI` for answer generation

## Endpoints

### `/ask`

Strict factual RAG over the document.

Use case:
- direct factual questions
- short answers grounded in document context

Examples:
- How many ECTS does this program have?
- What language is the program taught in?
- What English level is required?

### `/plan`

General step-by-step plan based on model knowledge.

Use case:
- broad questions
- general preparation guidance

### `/assistant`

Main hybrid assistant route.

Current logic:
- factual questions -> RAG first
- broader guidance questions -> contextual plan
- if RAG cannot answer -> fallback to contextual plan or general plan

## Current assistant behavior

The assistant currently tries to separate:

- document facts
- missing information
- general guidance

This is mainly used in `contextual_plan`.

Example behavior:
- factual question -> short answer from the document
- broader question -> contextual answer based on retrieved document context
- if the document is incomplete -> say what is missing and then give general guidance

## Example response modes

The assistant currently works with different response modes depending on the user request.

### `rag`

Used for direct factual questions.

Example:
- Question: `What English level is required?`
- Mode: `rag`
- Behavior: retrieves relevant chunks from the document and returns a short factual answer

### `contextual_plan`

Used for broader questions that need guidance, not just one fact.

Example:
- Question: `What documents do I need for this program?`
- Mode: `contextual_plan`
- Behavior:
  - extracts document-based facts
  - identifies missing information
  - adds practical guidance when the document is incomplete

### `fallback_plan`

Used when the document does not contain a direct answer and the assistant falls back to a more general plan.

Example:
- Question: broad question with weak or missing document support
- Mode: `fallback_plan`
- Behavior: gives general step-by-step guidance based on model knowledge

## What is already implemented

- strict RAG endpoint
- general planning endpoint
- hybrid `/assistant`
- intent-based retrieval shaping
- factual-first routing
- first working version of contextual planning

## Example questions

Factual:
- What language is the program taught in?
- What English level is required?
- How many ECTS does this program have?

Broader:
- What documents do I need for this program?
- How should I plan my studies for this program?
- What school subjects do I need before applying?

## Current limitations

- currently works with one PDF source
- no website crawling yet
- no multilingual support yet
- some broader answers are still too generic
- structure-heavy sections of the document can still be interpreted imperfectly
- not deployed

## Document source

This prototype currently uses one publicly available official curriculum PDF from the University of Klagenfurt.

The document remains the property of its original publisher and is included here only for educational / demonstration purposes in the context of this prototype.

## Why this project is interesting for me

I do not want to build just a chatbot that gives generic answers.

The more important part for me is:

- grounded answers
- separation between facts and assumptions
- reducing information chaos for the user
- building a useful assistant step by step

## Next steps

Planned next improvements:

- better contextual planning
- stronger handling of broad user questions
- support for multiple sources
- later: official university website ingestion
- later: multilingual support

## Tech stack

- Python
- FastAPI
- LangChain
- FAISS
- OpenAI API