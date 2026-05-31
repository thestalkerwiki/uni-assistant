# Uni Assistant

Uni Assistant is a backend prototype for AI-assisted university admission guidance.

The project helps future students understand university programmes, admission requirements, deadlines, and study structure based on official sources such as university websites and curriculum documents.

The long-term idea is to build an assistant for CIS / post-Soviet applicants who want to apply to European universities without fully relying on agencies.

## Project goals

This project currently has two main goals:

1. to serve as a portfolio project for IT job applications
2. to become a practical prototype of an AI assistant for university admission guidance

The focus is not only on answering questions, but also on separating official facts, missing information, and practical next steps.

## Current status

This is an early working prototype.

It is not production-ready, but it already supports:

- PDF-based RAG
- website ingestion
- multi-page web ingestion
- source type classification
- structured admission guidance
- multilingual response handling
- source-aware answers

## Core features

- FastAPI backend
- PDF document ingestion
- official university webpage ingestion
- multi-page web assistant
- text cleaning for noisy university webpages
- chunking with `RecursiveCharacterTextSplitter`
- OpenAI embeddings
- FAISS vector search
- ChatOpenAI answer generation
- web vectorstore caching
- intent-based routing
- answer mode detection
- source type classification:
  - `program_page`
  - `admission_page`
  - `deadline_page`
  - `language_page`
  - `unknown`

## Answer modes

The assistant supports several answer modes depending on the user request.

### `rag`

Used for direct factual questions.

Example:

```text
What language is the program taught in?
```

The assistant retrieves relevant chunks and returns a short factual answer.

### `evidence_summary`

Used when the user asks what is clearly stated on one or more pages.

Example:

```text
What requirements are clearly stated across these pages?
```

The assistant separates:

- clearly stated facts
- missing or unclear information

### `contextual_plan`

Used for broader guidance questions.

Example:

```text
What should I prepare before applying to this Master's programme?
```

The assistant returns:

- document-based facts
- missing information
- practical next steps

### `contextual_plan_fallback`

Used when the main structured response fails or returns empty output.

The fallback still answers using the retrieved source context instead of switching to an ungrounded general answer.

## Current architecture

Main pipeline:

```text
URLs / PDFs
→ loaders
→ text cleaning
→ chunks
→ embeddings
→ FAISS vectorstore
→ retrieval
→ source classification
→ prompt routing
→ LLM answer
→ structured response
```

## Main endpoints

### `/ask`

Strict factual RAG over the local PDF source.

### `/plan`

General step-by-step planning endpoint.

### `/assistant`

Hybrid assistant route for the local PDF source.

### `/web-ask`

Single-page web RAG endpoint.

### `/web-assistant`

Single-page web assistant with routing and structured answer modes.

### `/web-multi-assistant`

Multi-page web assistant.

This endpoint accepts several official university URLs and answers questions using all sources together.

Example source set:

- programme page
- admission page
- deadline page

## Example multi-page request

```json
{
  "urls": [
    "https://www.tugraz.at/studium/studienangebot/masterstudien/computational-social-systems",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassung-von-internationalen-studienwerberinnen-und-werbern/ueberblick-zulassung-von-internationalen-studienwerberinnen-und-werbern",
    "https://www.tugraz.at/studium/studieren-an-der-tu-graz/studieninteressierte/anmeldung-und-zulassung/zulassungsfristen"
  ],
  "question": "What should I prepare before applying to this Master's programme?"
}
```

Expected behavior:

- identify the programme page
- identify the admission page
- identify the deadline page
- extract programme facts
- extract admission facts
- extract deadlines
- return missing information and next steps

## Example output structure

```json
{
  "mode": "contextual_plan",
  "question": "What should I prepare before applying to this Master's programme?",
  "document_facts": [
    "The programme is Computational Social Systems.",
    "The degree is Master of Science (MSc).",
    "The programme has 120 ECTS.",
    "The duration is 4 semesters.",
    "The language of instruction is English."
  ],
  "missing_info": [
    "The exact accepted English language certificates are not stated in the retrieved context.",
    "The applicant's country-specific document requirements are not known."
  ],
  "answer": "1. Check whether your previous degree is relevant for the programme.\n2. Prepare the admission application documents.\n3. Check the correct deadline category.\n4. Prepare documents in the required language or format.",
  "sources": [
    "programme page URL",
    "admission page URL",
    "deadline page URL"
  ]
}
```

## Multilingual behavior

The assistant can detect the user request language and answer in that language.

For example, a user can ask in Russian while the source pages are in German or English.

Example:

```text
что я должен подготовить чтобы поступить на эту программу?
```

The assistant should answer in Russian while grounding the answer in the official source pages.

## Why source type classification matters

University information is often split across several pages.

For example:

- programme page: degree, ECTS, duration, language
- admission page: procedure, documents, applicant categories
- deadline page: application periods and deadlines

Source type classification helps the assistant understand where each piece of information comes from and prepares the project for future conflict handling between sources.

## Current limitations

- not deployed yet
- no frontend yet
- no user accounts or persistence
- no applicant profile yet
- no source-balanced retrieval yet
- some pages still contain navigation noise
- some answers may still miss available facts if retrieval does not select the right chunks
- not production-ready

## Next steps

Planned improvements:

- source-balanced retrieval across programme, admission, and deadline pages
- applicant profile
- conflict detection between official sources
- document checklist extraction
- simple demo UI
- short GIF / MP4 demo for portfolio
- later: user workspace and saved application roadmap

## Tech stack

- Python
- FastAPI
- LangChain
- FAISS
- OpenAI API
- Pydantic
- BeautifulSoup / bs4

## Project motivation

I do not want to build just a generic chatbot.

The main goals are:

- grounded answers
- clear distinction between facts and assumptions
- practical guidance for applicants
- reduction of information chaos
<<<<<<< HEAD
- step-by-step development of a useful assistant
=======
- step-by-step development of a useful assistant
>>>>>>> feature/multi-page-ingestion
