# uni-assistant
FastAPI-based RAG prototype for university admission guidance with contextual planning

# Uni Assistant

Uni Assistant is my backend prototype for university admission guidance.

The idea is to build an AI assistant that can help future students understand university programs, requirements, and study structure based on official documents.

Right now this is still an early version focused on one program and one PDF document.

## Project goal

This project has two main goals:

1. portfolio project for IT job applications  
2. practical prototype of an AI assistant for university admission guidance

Long term idea:
an assistant for CIS / post-Soviet applicants who want to apply to European universities without agencies.

## Current version

Current version works with:
- FastAPI
- one PDF source
- RAG
- FAISS vector search
- OpenAI model
- contextual planning for broader questions

At this stage the project is not production-ready.
It is a working prototype.

## Current architecture

Pipeline right now:

PDF -> chunks -> embeddings -> FAISS -> retrieval -> GPT answer

Main file:
- `app.py`

Main components:
- PDF loading with `PyPDFLoader`
- chunking with `RecursiveCharacterTextSplitter`
- embeddings with `OpenAIEmbeddings`
- vector search with `FAISS`
- answer generation with `ChatOpenAI`

## Endpoints

### `/ask`
Strict factual RAG over the document.

Use case:
- direct factual questions
- short document-based answers

Example:
- How many ECTS does this program have?
- What language is the program taught in?

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

This is especially used in `contextual_plan`.

Example behavior:
- factual question -> short answer from document
- broader question -> contextual answer based on retrieved document context
- if document is incomplete -> say what is missing and then give general guidance

## What is already implemented

- strict RAG endpoint
- general planning endpoint
- hybrid `/assistant`
- intent-based retrieval shaping
- factual-first routing
- first working version of contextual planning

## Current limitations

- only one PDF source
- no website crawling yet
- no multilingual support yet
- some broader answers are still too generic
- structure-heavy sections of the document can still be interpreted imperfectly
- not deployed

## Example questions

Factual:
- What language is the program taught in?
- What English level is required?
- How many ECTS does this program have?

Broader:
- What documents do I need for this program?
- How should I plan my studies for this program?
- What school subjects do I need before applying?

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
