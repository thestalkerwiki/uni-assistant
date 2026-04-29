import os
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = os.getenv(
    "USER_AGENT",
    "UniAssistantPrototype/1.0 (educational project)"
)

from fastapi import FastAPI
from pydantic import BaseModel, Field
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

app = FastAPI(title="Uni Assistant API")

WEB_VECTORSTORE_CACHE = {}

def is_factual_question(query: str) -> bool:
    lowered = query.lower().strip()

    factual_starts = [
        "how many",
        "how long",
        "what is",
        "what are",
        "what language",
        "what english level",
        "which",
        "when",
        "where",
        "does",
        "do",
        "is",
        "are",
        "can",
        "who"
    ]

    return any(lowered.startswith(start) for start in factual_starts)

def detect_answer_mode(query: str) -> str:
    lowered = query.lower().strip()

    if any(phrase in lowered for phrase in [
        "what should i",
        "how should i",
        "how do i prepare",
        "how should i prepare",
        "what should i prepare",
        "what should i focus on",
        "how should i plan",
        "what do i need to do",
        "what steps should i take",
        "how do i apply"
    ]):
        return "guidance_plan"

    if any(phrase in lowered for phrase in [
        "what does this page say about",
        "what requirements are clearly stated",
        "what requirements are stated",
        "what deadlines are mentioned",
        "what is clearly stated",
        "what is mentioned on this page",
        "what does the page mention"
    ]):
        return "evidence_summary"

    return "direct_answer"

def detect_query_intent(query: str) -> str:
    lowered = query.lower().strip()

    if any(word in lowered for word in [
        "deadline", "deadlines", "when is", "until when", "due date",
        "application deadline", "start date", "semester start"
    ]):
        return "deadline"

    if any(word in lowered for word in [
        "document", "documents", "certificate", "certificates",
        "transcript", "transcripts", "cv", "motivation letter",
        "proof", "proof of english", "language certificate"
    ]):
        return "documents"

    if any(word in lowered for word in [
        "admission", "apply", "application", "requirements",
        "eligible", "eligibility", "before applying",
        "how should i prepare", "what do i need before applying",
        "how to apply"
    ]):
        return "admission"
    
    
    if any(word in lowered for word in [
    "english level", "language requirement", "language requirements",
    "proof of english", "b2", "ielts", "toefl"
    ]):
        return "admission"

    if any(word in lowered for word in [
    "study plan", "plan my studies", "how should i plan my studies",
    "semester", "semesters", "ects", "curriculum",
    "minor", "minors", "elective", "electives",
    "bachelor thesis", "steop",
    "structure", "program structure", "study structure"
    ]):
        return "study_structure"
    
    if any(word in lowered for word in [
    "language of instruction", "taught in", "what language is the program taught in"
    ]):
        return "language"

    return "general"

def build_retrieval_query(user_query: str) -> str:
    intent = detect_query_intent(user_query)

    if intent == "documents":
        return (
            f"{user_query} "
            "admission requirements application documents required documents "
            "language requirements english b2 certificates transcript proof"
        )

    if intent == "admission":
        return (
            f"{user_query} "
            "admission requirements application eligibility english b2 "
            "zulassungsvoraussetzungen language requirements apply"
        )

    if intent == "deadline":
        return (
            f"{user_query} "
            "deadline application deadline semester start admission period dates"
        )
        
    if intent == "language":
        return (
            f"{user_query} "
            "fact box language of instruction english "
            "\"Language of instruction: English\" "
            "programme type duration ects structure degree prerequisites"
        )

    if intent == "study_structure":
        return (
            f"{user_query} "
            "curriculum 180 ects six semesters study duration "
            "required subjects guided electives minors bachelor thesis "
            "steop recommended study flow mobility window semester plan"
        )

    return user_query

def load_web_documents_from_url(url: str):
    loader = WebBaseLoader(
        url,
        header_template={
            "User-Agent": "UniAssistantPrototype/1.0 (educational project)"
        },
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                ["main", "article", "h1", "h2", "h3", "p", "li"]
            )
        }
    )
    docs = loader.load()

    if docs:
        print("DEBUG WEB TITLE:", docs[0].metadata.get("title"))
        print("DEBUG WEB SOURCE:", docs[0].metadata.get("source"))
        print("DEBUG WEB CONTENT PREVIEW:")
        print(docs[0].page_content[:1500])

    return docs

def build_web_vectorstore(url: str, embeddings):
    normalized_url = url.strip()

    if normalized_url in WEB_VECTORSTORE_CACHE:
        print(f"DEBUG web cache hit: {normalized_url}")
        return WEB_VECTORSTORE_CACHE[normalized_url]

    print(f"DEBUG web cache miss: {normalized_url}")

    docs = load_web_documents_from_url(normalized_url)

    for doc in docs:
        metadata = doc.metadata or {}
        title = metadata.get("title", "")
        source = metadata.get("source", "")

        header_parts = []
        if title:
            header_parts.append(f"Title: {title}")
        if source:
            header_parts.append(f"Source: {source}")

        if header_parts:
            doc.page_content = "\n".join(header_parts) + "\n\n" + doc.page_content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    WEB_VECTORSTORE_CACHE[normalized_url] = vectorstore
    return vectorstore

def extract_sources(results):
    sources = []

    for doc in results:
        metadata = doc.metadata or {}

        # PDF-style source
        page = metadata.get("page")
        if page is not None:
            sources.append(f"Page {page + 1}")
            continue

        # Web-style source
        title = metadata.get("title")
        source_url = metadata.get("source")

        if title and source_url:
            sources.append(f"{title} | {source_url}")
        elif source_url:
            sources.append(source_url)
        elif title:
            sources.append(title)
        else:
            sources.append("Source unknown")

    return list(dict.fromkeys(sources))

def build_context_from_docs(results):
    parts = []

    for doc in results:
        metadata = doc.metadata or {}
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        content = doc.page_content.strip()

        meta_block = []
        if title:
            meta_block.append(f"Title: {title}")
        if source:
            meta_block.append(f"Source: {source}")

        if meta_block:
            parts.append("\n".join(meta_block) + f"\nContent:\n{content}")
        else:
            parts.append(content)

    return "\n\n---\n\n".join(parts)

def ask_question(query, vectorstore, llm):
    retrieval_query = build_retrieval_query(query)
    intent = detect_query_intent(query)

    k = 8 if intent == "language" else 3
    results = vectorstore.similarity_search(retrieval_query, k=k)

    for i, doc in enumerate(results):
        print(f"DEBUG TOP CHUNK {i+1}:")
        print(doc.page_content[:500])
        print("---")

    context = build_context_from_docs(results)

    prompt = f"""
Answer ONLY based on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("DEBUG ask intent:", intent)
    print("DEBUG ask retrieval_query:", retrieval_query)
    print("DEBUG ask k:", k)

    sources = extract_sources(results)

    return {
        "mode": "rag",
        "question": query,
        "answer": response.content.strip(),
        "sources": sources
    }

def build_study_plan(user_request, llm):
    prompt = f"""
You are an expert university admission assistant.

Your job is to create a clear, practical step-by-step plan.

You are allowed to use general knowledge about studying abroad.

Requirements:
- Be practical
- Use numbered steps
- Each step should be actionable
- Keep it simple and clear
- If the request is too general, say what information is missing
- Then provide a helpful general plan anyway

User request:
{user_request}
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    if not text:
        text = (
            "I could not generate a detailed plan for this request yet. "
            "Please try rephrasing the question or ask a more specific question "
            "about admission, documents, deadlines, or study structure."
        )

    return {
        "mode": "plan",
        "request": user_request,
        "answer": text,
        "sources": []
    }
    
    
def build_evidence_summary(user_request, vectorstore, llm):
    intent = detect_query_intent(user_request)
    retrieval_query = build_retrieval_query(user_request)

    k = 6 if intent in {"deadline", "admission", "study_structure", "language"} else 4
    results = vectorstore.similarity_search(retrieval_query, k=k)

    context = build_context_from_docs(results)

    print("DEBUG summary intent:", intent)
    print("DEBUG summary retrieval_query:", retrieval_query)
    print("DEBUG summary k:", k)

    sources = extract_sources(results)

    prompt = f"""
You are an expert university admission assistant.

Your task is to answer the user's question by summarizing only what is clearly supported by the document context.

Rules:
- Answer the user's question directly.
- Prioritize extraction over advice.
- Do NOT turn the answer into a step-by-step plan unless the user explicitly asks for guidance.
- Separate clearly:
  1. what is explicitly stated
  2. what is not stated / missing
- Keep the answer concise and practical.
- Write in the same language as the user's question.

Document context:
{context}

User request:
{user_request}

Return the result in exactly this format:

DOCUMENT_FACTS:
- ...
- ...

MISSING_INFO:
- ...
- ...
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    print("DEBUG SUMMARY RAW RESPONSE:")
    print(repr(response.content))
    print("DEBUG SUMMARY STRIPPED TEXT:")
    print(repr(text))

    if not text:
        fallback = build_study_plan(user_request, llm)
        return {
            "mode": "fallback_plan",
            "request": user_request,
            "answer": fallback["answer"],
            "sources": sources
        }

    document_facts = []
    missing_info = []
    answer_parts = []

    section = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper() == "DOCUMENT_FACTS:":
            section = "facts"
            continue
        elif stripped.upper() == "MISSING_INFO:":
            section = "missing"
            continue

        if not stripped:
            continue

        if section == "facts":
            if stripped.startswith("-"):
                document_facts.append(stripped[1:].strip())
            else:
                document_facts.append(stripped)
        elif section == "missing":
            if stripped.startswith("-"):
                missing_info.append(stripped[1:].strip())
            else:
                missing_info.append(stripped)

    if document_facts:
        answer_parts.append("Clearly stated:")
        for fact in document_facts:
            answer_parts.append(f"- {fact}")

    if missing_info:
        answer_parts.append("")
        answer_parts.append("Not stated / unclear:")
        for item in missing_info:
            answer_parts.append(f"- {item}")

    answer = "\n".join(answer_parts).strip()

    if not answer:
        answer = text

    return {
        "mode": "evidence_summary",
        "question": user_request,
        "document_facts": document_facts,
        "missing_info": missing_info,
        "answer": answer,
        "sources": sources
    }
    
def build_contextual_plan(user_request, vectorstore, llm):
    intent = detect_query_intent(user_request)
    retrieval_query = build_retrieval_query(user_request)

    k = 6 if intent == "study_structure" else 4
    results = vectorstore.similarity_search(retrieval_query, k=k)

    context = build_context_from_docs(results)

    print("DEBUG contextual intent:", intent)
    print("DEBUG contextual retrieval_query:", retrieval_query)
    print("DEBUG contextual k:", k)

    sources = extract_sources(results)

    prompt = f"""
You are an expert university admission assistant.

Your task is to answer the user's request using the document context below.

You must produce:
1. DOCUMENT_FACTS -> only facts directly supported by the document context
2. MISSING_INFO -> only information needed for the user's request but NOT stated in the document context
3. PLAN -> a practical numbered plan

Strict rules:
- Use the document context as the primary source of truth.
- Include in DOCUMENT_FACTS only facts relevant to the user's specific request.
- Do NOT include unrelated facts, even if they appear in the document.
- Do NOT put something in MISSING_INFO if it is already stated in the document context.
- If the document does not fully answer the question, you may add general guidance in PLAN.
- Every step in PLAN that is based on general knowledge rather than the document must begin with:
  "General guidance:"
- Do NOT present general guidance as if it came from the document.
- Keep the answer practical, concise, and question-focused.
- Write in English only.

Document context:
{context}

User request:
{user_request}

Return the result in exactly this format:

DOCUMENT_FACTS:
- ...
- ...

MISSING_INFO:
- ...
- ...

PLAN:
1. ...
2. ...
3. ...

Extra instructions by intent:

If the question is about documents or admission:
- Focus on admission requirements, language requirements, and what the document does or does not specify.
- Do not invent application checklists if they are not stated.

If the question is about school background or eligibility:
- Do not invent required school subjects if they are not stated.

If the question is about study planning or program structure:
- Prefer facts about:
  - total ECTS
  - duration in semesters
  - study phases such as StEOP
  - required subjects
  - guided electives
  - minors / Nebenfächer
  - bachelor thesis
  - recommended study flow
  - mobility window
- If these are present in the context, do NOT list them under MISSING_INFO.
- Build a study plan in a sensible order:
  start of studies -> workload/progression -> guided electives/minors -> thesis -> mobility if relevant.
- Be especially careful with minors / Nebenfächer:
  - do not invent ECTS values
  - do not confuse item numbers (such as 12, 13, 14) with ECTS credits
  - if the context states the structure clearly, preserve it exactly

Prefer fewer but more accurate facts over many weak facts.
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    print("DEBUG CONTEXTUAL RAW RESPONSE:")
    print(repr(response.content))
    print("DEBUG CONTEXTUAL STRIPPED TEXT:")
    print(repr(text))

    if not text:
        fallback = build_study_plan(user_request, llm)
        return {
            "mode": "fallback_plan",
            "request": user_request,
            "answer": fallback["answer"],
            "sources": sources
        }

    document_facts = []
    missing_info = []
    answer = text

    section = None
    plan_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper() == "DOCUMENT_FACTS:":
            section = "facts"
            continue
        elif stripped.upper() == "MISSING_INFO:":
            section = "missing"
            continue
        elif stripped.upper() == "PLAN:":
            section = "plan"
            continue

        if not stripped:
            continue

        if section == "facts":
            if stripped.startswith("-"):
                document_facts.append(stripped[1:].strip())
            else:
                document_facts.append(stripped)

        elif section == "missing":
            if stripped.startswith("-"):
                missing_info.append(stripped[1:].strip())
            else:
                missing_info.append(stripped)

        elif section == "plan":
            plan_lines.append(stripped)

    if plan_lines:
        answer = "\n".join(plan_lines)

    return {
        "mode": "contextual_plan",
        "question": user_request,
        "document_facts": document_facts,
        "missing_info": missing_info,
        "answer": answer,
        "sources": sources
    }
    

# --- загрузка и подготовка базы один раз при старте ---
loader = PyPDFLoader("data/uni.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0,
    max_tokens=600
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="User question for Uni Assistant"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "what English level is required"
            }
        }
    }
    
class WebQuestionRequest(BaseModel):
    url: str
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.aau.at/en/studien/bachelor-digital-media-culture-and-communication/",
                "question": "What is the name of the program?"
            }
        }
    }

@app.get("/")
def root():
    return {"message": "Uni Assistant API is running"}


@app.post("/ask")
def ask(request: QuestionRequest):
    return ask_question(request.question, vectorstore, llm)

@app.post("/plan")
def build_plan(request: QuestionRequest):
    result = build_study_plan(request.question, llm)
    print("DEBUG PLAN RESPONSE:", result)
    return result
    
@app.post("/assistant")
def assistant(request: QuestionRequest):
    query = request.question.strip()
    intent = detect_query_intent(query)
    factual = is_factual_question(query)
    answer_mode = detect_answer_mode(query)

    print("DEBUG query:", query)
    print("DEBUG detected_intent:", intent)
    print("DEBUG is_factual:", factual)
    print("DEBUG answer_mode:", answer_mode)

    plan_intents = {"documents", "admission", "deadline", "study_structure", "language"}

    if factual and answer_mode == "direct_answer":
        print("DEBUG route: factual -> rag")
        rag_result = ask_question(query, vectorstore, llm)
        print("DEBUG rag_result:", rag_result)

        answer_text = rag_result["answer"].strip().lower()

        if "i don't know" in answer_text or "i do not know" in answer_text:
            if intent in plan_intents:
                print("DEBUG route: rag -> evidence_summary")
                return build_evidence_summary(query, vectorstore, llm)

            print("DEBUG route: fallback_plan")
            plan_result = build_study_plan(query, llm)
            plan_result["mode"] = "fallback_plan"
            return plan_result

        return rag_result

    if answer_mode == "evidence_summary":
        print("DEBUG route: evidence_summary")
        return build_evidence_summary(query, vectorstore, llm)

    if answer_mode == "guidance_plan":
        print("DEBUG route: contextual_plan")
        return build_contextual_plan(query, vectorstore, llm)

    print("DEBUG route: default rag")
    rag_result = ask_question(query, vectorstore, llm)
    print("DEBUG rag_result:", rag_result)

    answer_text = rag_result["answer"].strip().lower()

    if "i don't know" in answer_text or "i do not know" in answer_text:
        print("DEBUG route: fallback_plan")
        plan_result = build_study_plan(query, llm)
        plan_result["mode"] = "fallback_plan"
        return plan_result

    return rag_result

@app.post("/web-ask")
def web_ask(request: WebQuestionRequest):
    web_vectorstore = build_web_vectorstore(request.url, embeddings)
    return ask_question(request.question, web_vectorstore, llm)

@app.post("/web-assistant")
def web_assistant(request: WebQuestionRequest):
    web_vectorstore = build_web_vectorstore(request.url, embeddings)

    query = request.question.strip()
    intent = detect_query_intent(query)
    factual = is_factual_question(query)
    answer_mode = detect_answer_mode(query)

    print("DEBUG web query:", query)
    print("DEBUG web detected_intent:", intent)
    print("DEBUG web is_factual:", factual)
    print("DEBUG web answer_mode:", answer_mode)

    plan_intents = {"documents", "admission", "deadline", "study_structure", "language"}

    if factual and answer_mode == "direct_answer":
        print("DEBUG web route: factual -> rag")
        rag_result = ask_question(query, web_vectorstore, llm)
        print("DEBUG web rag_result:", rag_result)

        answer_text = rag_result["answer"].strip().lower()

        if "i don't know" in answer_text or "i do not know" in answer_text:
            if intent in plan_intents:
                print("DEBUG web route: rag -> evidence_summary")
                return build_evidence_summary(query, web_vectorstore, llm)

            print("DEBUG web route: fallback_plan")
            plan_result = build_study_plan(query, llm)
            plan_result["mode"] = "fallback_plan"
            return plan_result

        return rag_result

    if answer_mode == "evidence_summary":
        print("DEBUG web route: evidence_summary")
        return build_evidence_summary(query, web_vectorstore, llm)

    if answer_mode == "guidance_plan":
        print("DEBUG web route: contextual_plan")
        return build_contextual_plan(query, web_vectorstore, llm)

    print("DEBUG web route: default rag")
    rag_result = ask_question(query, web_vectorstore, llm)
    print("DEBUG web rag_result:", rag_result)

    answer_text = rag_result["answer"].strip().lower()

    if "i don't know" in answer_text or "i do not know" in answer_text:
        print("DEBUG web route: fallback_plan")
        plan_result = build_study_plan(query, llm)
        plan_result["mode"] = "fallback_plan"
        return plan_result

    return rag_result

@app.post("/web-preview")
def web_preview(request: WebQuestionRequest):
    docs = load_web_documents_from_url(request.url)

    if not docs:
        return {"error": "No documents loaded"}

    doc = docs[0]
    return {
        "source": doc.metadata.get("source"),
        "title": doc.metadata.get("title"),
        "language": doc.metadata.get("language"),
        "preview": doc.page_content[:1500]
    }