import os
import re
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Uni Assistant API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/demo")
def demo_page():
    return FileResponse("static/demo.html")

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
        # English
        "what should an applicant know",
        "summarize this programme",
        "summarize this program",
        "summarize this programme for an applicant",
        "summarize this program for an applicant",
        "programme overview",
        "program overview",
        "applicant overview",
        "what should i",
        "how should i",
        "how do i prepare",
        "how should i prepare",
        "what should i prepare",
        "what should i focus on",
        "how should i plan",
        "what do i need to do",
        "what steps should i take",
        "how do i apply",

        # Russian
        "что мне подготовить",
        "как мне подготовиться",
        "как подготовиться",
        "что нужно подготовить",
        "на чем сфокусироваться",
        "как спланировать",
        "что мне делать",
        "какие шаги",
        "что я должен подготовить",
        "что мне нужно подготовить",
        "что надо подготовить",
        "что подготовить",
        "как поступить",
        "чтобы поступить",

        # German
        "was sollte ein bewerber wissen",
        "was sollte eine bewerberin wissen",
        "fasse dieses studium zusammen",
        "fasse diesen studiengang zusammen",
        "überblick über das studium",
        "studienüberblick",
        "bewerberüberblick",
        "was soll ich vorbereiten",
        "wie soll ich mich vorbereiten",
        "wie bereite ich mich",
        "worauf soll ich achten",
        "was muss ich vorbereiten",
        "welche schritte"
    ]):
        return "guidance_plan"

    if any(phrase in lowered for phrase in [
        # English
        "what does this page say about",
        "what requirements are clearly stated",
        "what requirements are stated",
        "what deadlines are mentioned",
        "what is clearly stated",
        "what is mentioned on this page",
        "what does the page mention",

        # Russian
        "что говорится на этой странице",
        "что сказано на этой странице",
        "какие требования явно указаны",
        "какие требования указаны",
        "какие дедлайны указаны",
        "какие сроки указаны",
        "что явно указано",
        "что упомянуто на этой странице",

        # German
        "was steht auf dieser seite",
        "was sagt diese seite",
        "welche voraussetzungen sind",
        "welche anforderungen sind",
        "welche fristen werden",
        "welche deadlines werden",
        "was ist klar angegeben",
        "was wird auf dieser seite erwähnt"
    ]):
        return "evidence_summary"

    return "direct_answer"

def detect_query_intent(query: str) -> str:
    lowered = query.lower().strip()
    
    if any(phrase in lowered for phrase in [
        # English
        "programme overview",
        "program overview",
        "summarize this programme",
        "summarize this program",
        "what should an applicant know",
        "applicant overview",

        # German
        "studienüberblick",
        "überblick über das studium",
        "fasse dieses studium zusammen",
        "fasse diesen studiengang zusammen",
        "was sollte ein bewerber wissen",
        "was sollte eine bewerberin wissen"
    ]):
        return "overview"

    if any(word in lowered for word in [
    # English
        "language of instruction", "taught in",
        "what language is the program taught in",

    # Russian
        "язык обучения", "на каком языке", "преподается",
        "преподаётся", "какой язык программы",

    # German
        "unterrichtssprache", "sprache des studiums",
        "auf welcher sprache", "in welcher sprache",
        "wird der studiengang unterrichtet"
    ]):
        return "language"
    
    if any(word in lowered for word in [
        # English
        "deadline", "deadlines", "when is", "until when",
        "due date", "application deadline", "start date",
        "semester start", "admission period", "application period",

        # Russian
        "дедлайн", "дедлайны", "срок", "сроки",
        "до какого", "когда подавать", "период подачи",
        "срок подачи", "сроки подачи", "дата начала",
        "начало семестра",

        # German
        "frist", "fristen", "bewerbungsfrist",
        "bis wann", "zeitraum", "bewerbungszeitraum",
        "zulassungsfrist", "anmeldefrist", "semesterbeginn"
    ]):
        return "deadline"

    if any(word in lowered for word in [
        "document", "documents", "certificate", "certificates",
        "transcript", "transcripts", "cv", "motivation letter",
        "proof", "proof of english", "language certificate"
    ]):
        return "documents"

    if any(word in lowered for word in [
    # English
        "admission", "apply", "application", "requirements",
        "eligible", "eligibility", "before applying",
        "how should i prepare", "what do i need before applying",
        "how to apply",

    # Russian
        "поступление", "поступить", "как поступить", "чтобы поступить",
        "подать заявку", "заявка", "требования",
        "допуск", "перед подачей", "перед поступлением",
        "как подготовиться", "что нужно подготовить",
        "что я должен подготовить", "что мне нужно подготовить",
        "что надо подготовить", "что подготовить",

    # German
        "bewerbung", "zulassung", "voraussetzungen",
        "anforderungen", "bewerben", "einschreibung",
        "immatrikulation", "vor der bewerbung"
    ]):
        return "admission"
    
    
    if any(word in lowered for word in [
        # English
        "programme overview", "program overview",
        "summarize this programme", "summarize this program",
        "what should an applicant know",
        "applicant overview",

        # German
        "studienüberblick", "überblick über das studium",
        "fasse dieses studium zusammen",
        "fasse diesen studiengang zusammen",
        "was sollte ein bewerber wissen",
        "was sollte eine bewerberin wissen"
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

def detect_response_language(query: str) -> str:
    lowered = query.lower()

    # Russian / Cyrillic
    if any("а" <= char <= "я" or char == "ё" for char in lowered):
        return "Russian"

    # German markers
    german_markers = [
        "was ", "wie ", "welche", "welcher", "welches",
        "studiengang", "bewerbung", "zulassung",
        "voraussetzungen", "frist", "fristen",
        "unterrichtssprache", "dauer", "semester",
        "auf deutsch", "deutsch"
    ]

    if any(marker in lowered for marker in german_markers):
        return "German"

    return "English"

def build_retrieval_query(user_query: str) -> str:
    intent = detect_query_intent(user_query)
    
    if intent == "overview":
        return (
            f"{user_query} "
            "programme facts program facts degree ects duration language of instruction "
            "admission requirements required documents application form deadlines "
            "language requirements missing information next steps "
            "studiengang studium studiendauer ects-anrechnungspunkte abschluss "
            "unterrichtssprache zulassung voraussetzungen erforderliche unterlagen "
            "ansuchen um zulassung zulassungsfristen bewerbungsfrist"
        )

    if intent == "documents":
        return (
            f"{user_query} "
            "admission requirements application documents required documents "
            "language requirements english b2 certificates transcript proof"
        )

    if intent == "admission":
        return (
            f"{user_query} "
            # English
            "admission requirements application eligibility apply "
            "required documents application form legalisation legalization translation "
            "programme facts program facts degree ects duration language of instruction "
            "curriculum master bachelor deadlines application period "
            "tuition fees study start semester start "

            # German
            "zulassung voraussetzungen aufnahmeverfahren bewerbung "
            "erforderliche unterlagen benötigte unterlagen dokumente formular "
            "ansuchen um zulassung beglaubigung übersetzung legalisierung "
            "studiengang studium studiendauer ects-anrechnungspunkte abschluss "
            "unterrichtssprache curriculum masterstudium bachelorstudium "
            "zulassungsfristen bewerbungsfrist einreichfrist wintersemester sommersemester"
        )

    if intent == "deadline":
        return (
            f"{user_query} "
            "deadline application deadline semester start admission period dates"
        )
        
    if intent == "language":
        return (
            f"{user_query} "
            "language of instruction teaching language Unterrichtssprache "
            "program language programme language taught in "
            "language requirements CEFR english german deutsch "
            "programme type duration ects degree prerequisites"
        )

    if intent == "study_structure":
        return (
            f"{user_query} "
            "curriculum 180 ects six semesters study duration "
            "required subjects guided electives minors bachelor thesis "
            "steop recommended study flow mobility window semester plan"
        )

    return user_query

def clean_web_text(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings first, but keep line structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    noise_phrases = [
        "page navigation",
        "about the university",
        "organisation",
        "faculties",
        "working at university of graz",
        "developing solutions for the world of tomorrow",
        "research profile",
        "research questions",
        "research portal",
        "promoting research",
        "research transfer",
        "ethics in research",
        "commission for scientific integrity",
        "prospective students",
        "post-registration",
        "to improve support for screen readers",
        "to deactivate improved support",
        "go to overview of page sections",
        "begin of page section",
        "end of this page section",
        "go to contents",
        "go to position marker",
        "go to main navigation",
        "go to additional information",
        "go to page settings",
        "accesskey",
        "page sections",
        "page settings",
        "main navigation",
        "additional information",
        "accessibility declaration",
        "data protection declaration",
        "imprint",
        "sitemap",
        "web editors",
        "moodle",
        "unigrazonline",
        "zur hauptnavigation springen",
        "zum inhaltsbereich springen",
        "leichte sprache",
        "barrierefreiheit",

    ]

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        lowered = line.lower()

        # Remove obvious accessibility/navigation lines
        if any(phrase in lowered for phrase in noise_phrases):
            continue

        # Remove very short menu-like lines
        if len(line) <= 2:
            continue

        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)

    # Remove repeated leftover phrases
    text = re.sub(r"(Go to overview of page sections\s*)+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(End of this page section\.\s*)+", " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def classify_source_type(url: str, text: str) -> str:
    lowered_url = url.lower()
    lowered_text = text.lower()

    combined = lowered_url + " " + lowered_text[:3000]
    
    if any(marker in lowered_url for marker in [
        "zulassungsfristen",
        "deadlines",
        "dates-and-deadlines",
        "termine-fristen"
    ]):
        return "deadline_page"

    if any(marker in lowered_url for marker in [
        "zulassung-von-internationalen",
        "admission-bachelor",
        "applying",
        "admission",
        "bewerbung",
        "zulassung"
    ]):
        return "admission_page"
    
    if any(marker in lowered_url for marker in [
        "english-language",
        "language-requirements",
        "english-language-requirements"
    ]):
        return "language_page"

    if any(marker in lowered_url for marker in [
        "entry-requirements",
        "requirements"
    ]):
        return "admission_page"

    if any(marker in lowered_url for marker in [
        "fees-and-funding",
        "fees",
        "funding"
    ]):
        return "fees_page"

    if any(marker in lowered_url for marker in [
        "student-visa",
        "visa-guide",
        "visa"
    ]):
        return "visa_page"
    
    if any(marker in combined for marker in [
        "programme", "program", "degree", "ects", "curriculum",
        "duration", "study programme", "bachelor", "master",
        "studiengang", "studium", "curriculum", "regelstudienzeit"
    ]):
        return "program_page"

    if any(marker in combined for marker in [
        "language requirements", "language proof", "proof of language",
        "language certificate", "english b2", "german c1",
        "cefr", "ielts", "toefl",
        "sprachnachweis", "sprachkenntnisse", "unterrichtssprache"
    ]):
        return "language_page"

    return "unknown"

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

    for doc in docs:
        before_length = len(doc.page_content)
        doc.page_content = clean_web_text(doc.page_content)
        after_length = len(doc.page_content)

        source_url = doc.metadata.get("source", url)
        source_type = classify_source_type(source_url, doc.page_content)
        doc.metadata["source_type"] = source_type

        print(f"DEBUG web clean length: {before_length} -> {after_length}")
        print(f"DEBUG web source type: {source_type}")

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

def build_web_vectorstore_from_urls(urls: list[str], embeddings):
    normalized_urls = normalize_urls(urls)

    if not normalized_urls:
        raise ValueError("No valid URLs provided")

    cache_key = "multi::" + "|".join(sorted(normalized_urls))

    if cache_key in WEB_VECTORSTORE_CACHE:
        print(f"DEBUG web multi cache hit: {cache_key}")
        return WEB_VECTORSTORE_CACHE[cache_key]

    print(f"DEBUG web multi cache miss: {cache_key}")

    all_docs = []

    for url in normalized_urls:
        docs = load_web_documents_from_url(url)

        for doc in docs:
            metadata = doc.metadata or {}
            title = metadata.get("title", "")
            source = metadata.get("source", url)

            header_parts = []

            if title:
                header_parts.append(f"Title: {title}")

            if source:
                header_parts.append(f"Source: {source}")

            if header_parts:
                doc.page_content = "\n".join(header_parts) + "\n\n" + doc.page_content

            all_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    WEB_VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore

def normalize_urls(urls: list[str]) -> list[str]:
    cleaned_urls = []

    for url in urls:
        normalized_url = url.strip()

        if not normalized_url:
            continue

        if normalized_url not in cleaned_urls:
            cleaned_urls.append(normalized_url)

    return cleaned_urls

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
        source_type = metadata.get("source_type", "")
        content = doc.page_content.strip()

        meta_block = []
        if title:
            meta_block.append(f"Title: {title}")
        if source:
            meta_block.append(f"Source: {source}")
        if source_type:
            meta_block.append(f"Source type: {source_type}")

        if meta_block:
            parts.append("\n".join(meta_block) + f"\nContent:\n{content}")
        else:
            parts.append(content)

    return "\n\n---\n\n".join(parts)

def score_doc_for_source_type(doc, source_type: str) -> int:
    """
    Simple keyword scoring to prefer more useful chunks
    inside the same source_type.
    """
    text = doc.page_content.lower()

    keywords_by_source_type = {
        "program_page": [
            "studiendauer",
            "ects-anrechnungspunkte",
            "abschluss",
            "master of science",
            "bachelor of arts",
            "unterrichtssprache",
            "language of instruction",
            "duration",
            "ects",
            "degree",
            "computational social systems",
        ],
        "admission_page": [
            "ansuchen um zulassung",
            "unterlagen",
            "required documents",
            "documents",
            "zulassung",
            "aufnahmeverfahren",
            "legalisation",
            "legalization",
            "beglaubigen",
            "übersetzen",
            "translation",
            "vfs global",
        ],
        "deadline_page": [
            "masterstudien",
            "master studies",
            "master's programmes",
            "eu/ewr",
            "eu ewr",
            "drittstaat",
            "drittstaaten",
            "third-country",
            "wintersemester",
            "sommersemester",
            "winter semester",
            "summer semester",
            "15. oktober",
            "15. august",
            "15. märz",
            "15. jänner",
            "15 october",
            "15 august",
            "15 march",
            "15 january",
        ],
        "language_page": [
            "sprachnachweis",
            "sprachkenntnisse",
            "english proficiency",
            "proof of english",
            "cefr",
            "ielts",
            "toefl",
        ],
    }

    keywords = keywords_by_source_type.get(source_type, [])
    score = 0

    for keyword in keywords:
        if keyword in text:
            score += 1

    return score

def search_by_source_type(vectorstore, query: str, source_type: str, k: int, fetch_k: int = 30):
    """
    Search broadly, manually filter by source_type,
    then rank chunks by source-specific keyword score.
    """
    results = vectorstore.similarity_search(query, k=fetch_k)

    filtered = []
    for doc in results:
        metadata = doc.metadata or {}
        if metadata.get("source_type") == source_type:
            filtered.append(doc)

    ranked = sorted(
        filtered,
        key=lambda doc: score_doc_for_source_type(doc, source_type),
        reverse=True
    )

    selected = ranked[:k]

    print(f"DEBUG source search: {source_type} -> {len(selected)} chunks")

    for i, doc in enumerate(selected):
        score = score_doc_for_source_type(doc, source_type)
        print(f"DEBUG source ranked {source_type} #{i + 1}, score={score}")
        print(doc.page_content[:300])
        print("---")

    return selected

def balanced_similarity_search(vectorstore, query: str, intent: str, default_k: int = 4):
    """
    Source-balanced retrieval for multi-page web assistant.

    Uses source-specific queries and manual filtering by source_type.
    """
    
    if intent == "overview":
        source_type_targets = [
            ("program_page", 2),
            ("admission_page", 2),
            ("deadline_page", 1),
            ("language_page", 1),
            ("fees_page", 1),
            ("visa_page", 1),
        ]
    elif intent == "admission":
        source_type_targets = [
            ("program_page", 2),
            ("admission_page", 2),
            ("deadline_page", 1),
        ]   
    elif intent == "deadline":
        source_type_targets = [
            ("deadline_page", 4),
            ("program_page", 1),
            ("admission_page", 1),
        ]
    elif intent == "language":
        source_type_targets = [
            ("program_page", 2),
            ("language_page", 2),
            ("admission_page", 1),
        ]
    elif intent == "study_structure":
        source_type_targets = [
            ("program_page", 5),
            ("admission_page", 1),
        ]
    else:
        source_type_targets = [
            ("program_page", 2),
            ("admission_page", 1),
            ("deadline_page", 1),
        ]

    source_queries = {
        "program_page": (
            "programme facts program facts study programme degree ECTS duration language of instruction "
            "study start semester curriculum tuition fees programme profile "
            "Studiendauer ECTS-Anrechnungspunkte Abschluss Unterrichtssprache "
            "Studiengang Studium Masterstudium Bachelorstudium Curriculum Studienbeginn "
            "Master of Science Bachelor of Arts Computational Social Systems"
        ),
        "admission_page": (
            "admission requirements application form required documents eligibility procedure "
            "legalisation legalization translation certificates proof application steps "
            "Zulassung Voraussetzungen Aufnahmeverfahren Bewerbung Ansuchen um Zulassung "
            "erforderliche Unterlagen benötigte Unterlagen Dokumente Nachweise "
            "Beglaubigung Übersetzung Legalisierung länderspezifische Informationen"
        ),
        "deadline_page": (
            "deadlines application period admission period dates winter semester summer semester "
            "Bachelor studies Master studies Master's programmes EU EEA third-country "
            "Zulassungsfristen Bewerbungsfrist Einreichfrist Fristen Wintersemester Sommersemester "
            "Bachelorstudien Masterstudien EU EWR Drittstaaten "
            "1 Mai 15 August 15 Oktober 1 Dezember 15 Jänner 15 März"
        ),
        "language_page": (
            "language requirements proof of English English proficiency CEFR IELTS TOEFL "
            "Sprachnachweis Sprachkenntnisse Englisch Deutsch"
        ),
    }

    selected = []
    seen_contents = set()

    for source_type, k in source_type_targets:
        source_query = source_queries.get(source_type, query)

        fetch_k = 50 if source_type == "deadline_page" else 30

        results = search_by_source_type(
            vectorstore=vectorstore,
            query=source_query,
            source_type=source_type,
            k=k,
            fetch_k=fetch_k
        )
        for doc in results:
            content_key = doc.page_content[:300]
            if content_key not in seen_contents:
                selected.append(doc)
                seen_contents.add(content_key)

    # Fallback: fill missing context with general similarity search
    if len(selected) < default_k:
        fallback_results = vectorstore.similarity_search(query, k=default_k)

        for doc in fallback_results:
            content_key = doc.page_content[:300]
            if content_key not in seen_contents:
                selected.append(doc)
                seen_contents.add(content_key)

    print("DEBUG balanced retrieval selected:")
    for i, doc in enumerate(selected):
        source_type = (doc.metadata or {}).get("source_type", "unknown")
        source = (doc.metadata or {}).get("source", "unknown")
        print(f"DEBUG BALANCED CHUNK {i + 1}: {source_type} | {source}")
        print(doc.page_content[:400])
        print("---")

    return selected

def ask_question(query, vectorstore, llm):
    retrieval_query = build_retrieval_query(query)
    intent = detect_query_intent(query)
    response_language = detect_response_language(query)

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

Rules:
- Answer directly and concisely.
- Write the final answer in {response_language}.
- If you say "I don't know", write it in {response_language}.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("DEBUG ask intent:", intent)
    print("DEBUG ask retrieval_query:", retrieval_query)
    print("DEBUG ask k:", k)
    print("DEBUG response_language:", response_language)

    sources = extract_sources(results)

    return {
        "mode": "rag",
        "question": query,
        "answer": response.content.strip(),
        "sources": sources
    }

def build_study_plan(user_request, llm):
    response_language = detect_response_language(user_request)
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
- Write the final answer in {response_language}.

User request:
{user_request}
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    if not text:
        if response_language == "Russian":
            text = (
            "Я пока не смог сформировать подробный план для этого запроса. "
            "Попробуй переформулировать вопрос или спросить конкретнее про admission, documents, deadlines или study structure."
        )
        elif response_language == "German":
            text = (
            "Ich konnte für diese Anfrage noch keinen detaillierten Plan erstellen. "
            "Bitte formuliere die Frage genauer oder frage konkreter zu admission, documents, deadlines oder study structure."
        )
        else:
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
    response_language = detect_response_language(user_request)
    retrieval_query = build_retrieval_query(user_request)

    k = 6 if intent in {"deadline", "admission", "study_structure", "language"} else 4
    results = balanced_similarity_search(
        vectorstore=vectorstore,
        query=retrieval_query,
        intent=intent,
        default_k=6
    )
    
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
- Extract concrete facts with values whenever available: dates, deadlines, ECTS, duration, degree, language, required forms, document language, applicant categories.
- Avoid vague observations such as "students should check..." unless that is the only information stated.
- Prefer "field: value" style facts when possible.
- Do NOT turn the answer into a step-by-step plan unless the user explicitly asks for guidance.
- Separate clearly:
  1. what is explicitly stated
  2. what is not stated / missing
- Keep the answer concise and practical.
- Write the final answer in {response_language}.

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
    
def build_programme_overview(user_request, vectorstore, llm):
    response_language = detect_response_language(user_request)

    results = balanced_similarity_search(
        vectorstore=vectorstore,
        query=user_request,
        intent="overview",
        default_k=4
    )

    context = build_context_from_docs(results)
    sources = extract_sources(results)

    prompt = f"""
    You are an expert university admission assistant.

    Use ONLY the document context below.
    Create a concise applicant-facing overview.
    Do not invent missing facts.

    Write the answer in {response_language}.

    Use this exact structure:

    Programme
    - Name:
    - Degree:
    - Duration:
    - ECTS:
    - Language:

    Admission
    - Clearly stated admission facts:
    - Required form or procedure:

    Documents
    - Required or mentioned documents:
    - Translation/legalisation rules:

    Deadlines
    - Clearly stated deadlines or application periods:

    Missing information
    - Important information not stated in the provided context:

    Next steps
    1. ...
    2. ...
    3. ...

    Rules:
    - If a field is not stated, write "Not stated in the provided context."
    - Prefer concrete values over vague summaries.
    - Keep country-specific rules short and mark them as conditional.
    - Keep the answer practical and easy to scan.

    Document context:
    {context}

    User request:
    {user_request}
    """

    response = llm.invoke(prompt)
    text = response.content.strip()

    print("DEBUG OVERVIEW RAW RESPONSE:")
    print(repr(response.content))
    print("DEBUG OVERVIEW STRIPPED TEXT:")
    print(repr(text))

    if not text:
        text = (
            "I could not generate a programme overview from the provided sources. "
            "Try asking specifically about programme facts, admission requirements, documents, or deadlines."
        )

    return {
        "mode": "programme_overview",
        "question": user_request,
        "answer": text,
        "sources": sources
    }
    
def build_contextual_plan(user_request, vectorstore, llm):
    intent = detect_query_intent(user_request)
    response_language = detect_response_language(user_request)
    retrieval_query = build_retrieval_query(user_request)

    k = 6 if intent == "study_structure" else 4
    results = balanced_similarity_search(
        vectorstore=vectorstore,
        query=user_request,
        intent=intent,
        default_k=k
    )

    context = build_context_from_docs(results)

    print("DEBUG contextual intent:", intent)
    print("DEBUG contextual retrieval_query:", retrieval_query)
    print("DEBUG contextual k:", k)

    sources = extract_sources(results)

    prompt = f"""
You are an expert university admission assistant.

Answer the user's request using ONLY the document context below as the factual basis.

Return the answer in exactly this structure:

DOCUMENT_FACTS:
- facts that are directly supported by the context
- only include facts relevant to the user's request

MISSING_INFO:
- information that would be needed to fully answer the request but is not stated in the context

PLAN:
1. practical next step
2. practical next step
3. practical next step

Rules:
- Do not invent facts.
- Do not include unrelated facts.
- In DOCUMENT_FACTS, prefer concrete extracted facts with values: degree, ECTS, duration, language, deadlines, required documents, admission requirements, procedures, exceptions.
- Avoid vague advice-like statements in DOCUMENT_FACTS, such as "you should check..." or "some programmes may...".
- If the context contains numbers, dates, ECTS, semesters, language levels, or named requirements, include them when relevant.
- If the user asks about a specific programme and programme facts are available, include the programme name and key programme facts.
- If something is not stated, put it in MISSING_INFO.
- If a PLAN step uses general advice rather than the document context, start it with "General guidance:".
- Keep the answer concise and practical.
- Write the final answer in {response_language}.
- Prioritize facts by relevance for the average applicant.
- In DOCUMENT_FACTS, include the most generally relevant facts first.
- Put country-specific or conditional rules after general programme/admission facts.
- If a fact applies only to a specific group, clearly mark it as conditional.
- Do not let conditional exceptions dominate the answer.
- Prefix each DOCUMENT_FACTS item with one of:
  [general] for facts relevant to most applicants
  [conditional] for facts relevant only to specific countries, backgrounds, or situations
  [contact] for contact details
- Use [general] facts before [conditional] and [contact] facts.
- In PLAN, do not include conditional steps as main steps unless they are likely relevant to the user.
- If a step only applies to a specific country or applicant group, add it as a conditional note, not as a main step.
- PLAN should contain only broadly relevant next steps.
- Country-specific rules should go into DOCUMENT_FACTS as [conditional], not into PLAN, unless the user explicitly says they are from that country.

Document context:
{context}

User request:
{user_request}
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    print("DEBUG CONTEXTUAL RAW RESPONSE:")
    print(repr(response.content))
    print("DEBUG CONTEXTUAL STRIPPED TEXT:")
    print(repr(text))

    if not text:
        fallback_prompt = f"""
You are an expert university admission assistant.

The previous structured response was empty.
Use the document context below and answer the user's request directly.

Rules:
- Use only the document context as factual basis.
- If something is not stated, say that it is not stated.
- Give a short practical plan.
- Keep the answer concise.
- Write the final answer in {response_language}.
- Mark general advice with "General guidance:".

Document context:
{context}

User request:
{user_request}
"""

        fallback_response = llm.invoke(fallback_prompt)
        fallback_text = fallback_response.content.strip()

        if not fallback_text:
            fallback_text = (
                "I could not generate a grounded answer from the provided sources. "
                "Please try asking a more specific question about admission, documents, deadlines, or language requirements."
            )

        return {
            "mode": "contextual_plan_fallback",
            "question": user_request,
            "document_facts": [],
            "missing_info": [],
            "answer": fallback_text,
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

    answer_parts = []

    MAX_GENERAL_FACTS = 6
    MAX_CONDITIONAL_FACTS = 2
    MAX_CONTACT_FACTS = 2
    MAX_MISSING_INFO = 5
    MAX_PLAN_LINES = 4

    if document_facts:
        general_facts = []
        conditional_facts = []
        contact_facts = []
        other_facts = []

        for fact in document_facts:
            stripped_fact = fact.strip()
            lowered_fact = stripped_fact.lower()

            if lowered_fact.startswith("[conditional]"):
                conditional_facts.append(stripped_fact[len("[conditional]"):].strip())
            elif lowered_fact.startswith("[contact]"):
                contact_facts.append(stripped_fact[len("[contact]"):].strip())
            elif lowered_fact.startswith("[general]"):
                general_facts.append(stripped_fact[len("[general]"):].strip())
            else:
                other_facts.append(stripped_fact)

        main_facts = (general_facts + other_facts)[:MAX_GENERAL_FACTS]

        if main_facts:
            answer_parts.append("What I found:")
            for fact in main_facts:
                answer_parts.append(f"- {fact}")

        if conditional_facts:
            answer_parts.append("")
            answer_parts.append("Conditional notes:")
            for fact in conditional_facts[:MAX_CONDITIONAL_FACTS]:
                answer_parts.append(f"- {fact}")

        if contact_facts:
            answer_parts.append("")
            answer_parts.append("Contacts:")
            for fact in contact_facts[:MAX_CONTACT_FACTS]:
                answer_parts.append(f"- {fact}")

    if missing_info:
        answer_parts.append("")
        answer_parts.append("What is still unclear:")
        for item in missing_info[:MAX_MISSING_INFO]:
            answer_parts.append(f"- {item}")

    if plan_lines:
        answer_parts.append("")
        answer_parts.append("What you should do next:")
        answer_parts.extend(plan_lines[:MAX_PLAN_LINES])

    if answer_parts:
        answer = "\n".join(answer_parts)

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
    max_tokens=900
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
    
class WebMultiQuestionRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="List of official university page URLs to use as sources"
    )
    question: str = Field(
        ...,
        min_length=1,
        description="User question about the provided sources"
    )

class WebPreviewRequest(BaseModel):
    url: str

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

@app.post("/web-multi-assistant")
def web_multi_assistant(request: WebMultiQuestionRequest):
    web_vectorstore = build_web_vectorstore_from_urls(request.urls, embeddings)

    query = request.question.strip()
    intent = detect_query_intent(query)
    factual = is_factual_question(query)
    answer_mode = detect_answer_mode(query)

    print("DEBUG web multi urls:", normalize_urls(request.urls))
    print("DEBUG web multi query:", query)
    print("DEBUG web multi detected_intent:", intent)
    print("DEBUG web multi is_factual:", factual)
    print("DEBUG web multi answer_mode:", answer_mode)

    plan_intents = {"documents", "admission", "deadline", "study_structure", "language"}

    if factual and answer_mode == "direct_answer":
        print("DEBUG web multi route: factual -> rag")
        rag_result = ask_question(query, web_vectorstore, llm)
        print("DEBUG web multi rag_result:", rag_result)

        answer_text = rag_result["answer"].strip().lower()

        if "i don't know" in answer_text or "i do not know" in answer_text:
            if intent in plan_intents:
                print("DEBUG web multi route: rag -> evidence_summary")
                return build_evidence_summary(query, web_vectorstore, llm)

            print("DEBUG web multi route: fallback_plan")
            plan_result = build_study_plan(query, llm)
            plan_result["mode"] = "fallback_plan"
            return plan_result

        return rag_result

    if answer_mode == "evidence_summary":
        print("DEBUG web multi route: evidence_summary")
        return build_evidence_summary(query, web_vectorstore, llm)

    if answer_mode == "guidance_plan":
        if intent == "overview":
            print("DEBUG web multi route: programme_overview")
            return build_programme_overview(query, web_vectorstore, llm)

        print("DEBUG web multi route: contextual_plan")
        return build_contextual_plan(query, web_vectorstore, llm)

    print("DEBUG web multi route: default rag")
    rag_result = ask_question(query, web_vectorstore, llm)
    print("DEBUG web multi rag_result:", rag_result)

    answer_text = rag_result["answer"].strip().lower()

    if "i don't know" in answer_text or "i do not know" in answer_text:
        print("DEBUG web multi route: fallback_plan")
        plan_result = build_study_plan(query, llm)
        plan_result["mode"] = "fallback_plan"
        return plan_result

    return rag_result

@app.post("/web-preview")
def web_preview(request: WebPreviewRequest):
    docs = load_web_documents_from_url(request.url)

    if not docs:
        return {
            "source": request.url,
            "title": None,
            "language": None,
            "preview": ""
        }

    doc = docs[0]

    return {
        "source": doc.metadata.get("source"),
        "title": doc.metadata.get("title"),
        "language": doc.metadata.get("language"),
        "preview": doc.page_content[:1500]
    }
    
    