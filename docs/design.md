# Uni Assistant — Design Notes

## Problem

University admission information is often fragmented across multiple official sources:

- programme pages
- admission pages
- deadline pages
- curriculum PDFs
- language requirement pages
- country-specific document rules

Applicants often struggle to understand what is clearly stated, what is missing, and what they should do next.

## Goal

Build a backend assistant that can read official university sources and provide structured, source-aware guidance for applicants.

The assistant should not simply generate generic advice. It should separate:

- official facts
- missing or unclear information
- practical next steps

## Current architecture

```text
URL / PDF
→ loader
→ text cleaner
→ chunking
→ embeddings
→ FAISS vectorstore
→ retrieval
→ source type classification
→ prompt routing
→ LLM answer
→ structured response
```

## Source types

The current prototype classifies web sources into:

- `program_page`
- `admission_page`
- `deadline_page`
- `language_page`
- `unknown`

This allows the assistant to reason about where information comes from.

Example:

- programme facts should usually come from a programme page
- deadlines should usually come from a deadline page
- document procedures should usually come from an admission page

## Answer modes

### `direct_answer`

Short factual answer.

Example:

```text
What language is the programme taught in?
```

### `evidence_summary`

Used when the user asks what is clearly stated in the available sources.

Example:

```text
What requirements are clearly stated across these pages?
```

The assistant should return clearly stated facts and missing or unclear information.

### `guidance_plan`

Used when the user asks what to do or prepare.

Example:

```text
What should I prepare before applying to this Master's programme?
```

The assistant should return source-based facts, missing information, and practical next steps.

## Why structured output?

Applicants need clarity, not only raw answers.

The assistant returns:

- `DOCUMENT_FACTS`
- `MISSING_INFO`
- `PLAN`

This structure makes it easier to understand what is officially supported and what still needs checking.

## Multi-page source sets

The current web assistant can work with multiple official URLs at once.

A typical source set can contain:

- a programme page
- an admission page
- a deadline page

This is important because university information is rarely located on one page only.

Example:

```text
Programme page → degree, ECTS, duration, teaching language
Admission page → application procedure, required forms, applicant categories
Deadline page → application periods and deadlines
```

## Multilingual behavior

The assistant can detect the user request language and answer in that language.

This is important because the intended users may ask questions in Russian while official university sources are written in German or English.

Example:

```text
User question: что я должен подготовить чтобы поступить на эту программу?
Source pages: German / English university pages
Assistant answer: Russian, grounded in the official sources
```

## Current limitation

The next important technical improvement is source-balanced retrieval.

Currently, retrieval may select too many chunks from one source type and miss useful facts from another source type.

Future retrieval should deliberately include relevant chunks from:

- `program_page`
- `admission_page`
- `deadline_page`

This should make answers more stable and more fact-rich.

## Roadmap

1. Source-balanced retrieval
2. Applicant profile
3. Conflict-aware source comparison
4. Document checklist extraction
5. Minimal web UI
6. Portfolio GIF / MP4 demo
7. User workspace and saved application roadmap

## Design principle

The assistant should feel simple and accessible for users who do not want to manually search university websites or translate complex admission pages.

However, it should remain professionally trustworthy:

- facts must be grounded in sources
- missing information should be clearly marked
- general guidance should not be presented as official fact
- contacting the university should be a fallback, not the first answer when enough evidence is available
