from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI(title="Uni Assistant API")

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

    if intent == "study_structure":
        return (
            f"{user_query} "
            "curriculum 180 ects six semesters study duration "
            "required subjects guided electives minors bachelor thesis "
            "steop recommended study flow mobility window semester plan"
        )

    return user_query

def ask_question(query, vectorstore, llm):
    retrieval_query = build_retrieval_query(query)
    results = vectorstore.similarity_search(retrieval_query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Answer ONLY based on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    
    print("DEBUG ask intent:", detect_query_intent(query))
    print("DEBUG ask retrieval_query:", retrieval_query)

    sources = []
    for doc in results:
        page = doc.metadata.get("page", "unknown")
        if page != "unknown":
            sources.append(f"Page {page + 1}")
        else:
            sources.append("Page unknown")

    sources = list(dict.fromkeys(sources))

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

    return {
        "mode": "plan",
        "request": user_request,
        "answer": response.content.strip(),
        "sources": []
    }
    
def build_contextual_plan(user_request, vectorstore, llm):
    intent = detect_query_intent(user_request)
    retrieval_query = build_retrieval_query(user_request)

    k = 6 if intent == "study_structure" else 4
    results = vectorstore.similarity_search(retrieval_query, k=k)

    context = "\n\n".join([doc.page_content for doc in results])

    print("DEBUG contextual intent:", intent)
    print("DEBUG contextual retrieval_query:", retrieval_query)
    print("DEBUG contextual k:", k)

    sources = []
    for doc in results:
        page = doc.metadata.get("page", "unknown")
        if page != "unknown":
            sources.append(f"Page {page + 1}")
        else:
            sources.append("Page unknown")

    sources = list(dict.fromkeys(sources))

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
    model="gpt-4.1-mini",
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

    print("DEBUG query:", query)
    print("DEBUG detected_intent:", intent)
    print("DEBUG is_factual:", factual)

    plan_intents = {"documents", "admission", "deadline", "study_structure"}

    # 1. If the question is factual, try RAG first
    if factual:
        print("DEBUG route: factual -> rag")
        rag_result = ask_question(query, vectorstore, llm)
        print("DEBUG rag_result:", rag_result)

        answer_text = rag_result["answer"].strip().lower()

        if "i don't know" in answer_text or "i do not know" in answer_text:
            if intent in plan_intents:
                print("DEBUG route: rag -> contextual_plan")
                contextual_result = build_contextual_plan(query, vectorstore, llm)

                has_facts = len(contextual_result["document_facts"]) > 0
                has_sources = len(contextual_result["sources"]) > 0

                if has_facts or has_sources:
                    return contextual_result

            print("DEBUG route: fallback_plan")
            plan_result = build_study_plan(query, llm)
            plan_result["mode"] = "fallback_plan"
            return plan_result

        print("DEBUG route: rag")
        return rag_result

    # 2. If it is not factual, but is a planning/high-guidance intent -> contextual plan
    if intent in plan_intents:
        print("DEBUG route: contextual_plan")
        return build_contextual_plan(query, vectorstore, llm)

    # 3. Otherwise default to rag
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