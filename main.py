from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()


def ask_question(query, vectorstore, llm):
    results = vectorstore.similarity_search(query, k=3)

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

    sources = []
    for doc in results:
        page = doc.metadata.get("page", "unknown")
        if page != "unknown":
            sources.append(f"Page {page + 1}")
        else:
            sources.append("Page unknown")

    sources = list(dict.fromkeys(sources))  # убираем дубликаты, сохраняя порядок

    return {
        "question": query,
        "answer": response.content,
        "sources": sources
    }


loader = PyPDFLoader("data/uni.pdf")
docs = loader.load()

print("Pages:", len(docs))
print(docs[0].page_content[:300])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))
print(chunks[0].page_content)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector DB готова")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=200
)

queries = [
    "How many ECTS does the program have?",
    "How long does the program last?",
    "What language is the program taught in?",
    "What English level is required for admission?"
]

for q in queries:
    result = ask_question(q, vectorstore, llm)

    print("\n" + "=" * 70)
    print("QUESTION:", result["question"])
    print("ANSWER:", result["answer"])
    print("SOURCES:", ", ".join(result["sources"]))