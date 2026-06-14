"""
Specialist Agents:
- SummaryAgent    — structured document summaries
- ComparisonAgent — cross-document comparison tables
- AnalystAgent    — analytical reasoning, pros/cons, trends
- GeneralAgent    — general knowledge fallback
"""

from __future__ import annotations
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ── Summary Agent ─────────────────────────────────────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""You are an expert summariser.
Create a structured summary of the following document content.

Include:
- **Key Topics** (bullet list)
- **Main Arguments / Findings**
- **Important Data / Numbers** (if any)
- **Conclusions**

Document content:
{context}

User request: {question}

Produce a well-structured markdown summary:""")


class SummaryAgent:
    name = "summary_agent"
    description = "Summarises documents into structured formats"

    def __init__(self, model: str, base_url: str, retriever: Any | None = None):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)
        self.retriever = retriever

    def set_retriever(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> tuple[str, list[dict]]:
        if not self.retriever:
            return "No documents available to summarise.", []

        docs = self.retriever.invoke(query, config={"k": 6})
        if not docs:
            docs = self.retriever.invoke("summary overview")

        context = "\n\n".join(d.page_content for d in docs[:6])
        chain = SUMMARY_PROMPT | self.llm
        response = chain.invoke({"context": context, "question": query})

        sources = [
            {"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")}
            for d in docs[:3]
        ]
        return response.content, sources

    def stream(self, query: str):
        if not self.retriever:
            yield "No documents available.", []
            return

        docs = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs[:6])
        chain = SUMMARY_PROMPT | self.llm
        sources = [{"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")} for d in docs[:3]]

        for chunk in chain.stream({"context": context, "question": query}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token, None
        yield "", sources


# ── Comparison Agent ──────────────────────────────────────────────────────────

COMPARISON_PROMPT = ChatPromptTemplate.from_template("""You are an expert at comparing
information across documents. Use the provided context to compare items clearly.

Structure your response as:
- A brief introduction
- A comparison table (markdown) where appropriate
- Key differences / similarities
- Your assessment / recommendation

Context from documents:
{context}

Comparison question: {question}

Provide a thorough comparison:""")


class ComparisonAgent:
    name = "comparison_agent"
    description = "Compares information across documents or topics"

    def __init__(self, model: str, base_url: str, retriever: Any | None = None):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.1)
        self.retriever = retriever

    def set_retriever(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> tuple[str, list[dict]]:
        if not self.retriever:
            return "No documents available for comparison.", []

        docs = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = COMPARISON_PROMPT | self.llm
        response = chain.invoke({"context": context, "question": query})
        sources = [{"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")} for d in docs[:3]]
        return response.content, sources

    def stream(self, query: str):
        if not self.retriever:
            yield "No documents available.", []
            return

        docs = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = COMPARISON_PROMPT | self.llm
        sources = [{"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")} for d in docs[:3]]

        for chunk in chain.stream({"context": context, "question": query}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token, None
        yield "", sources


# ── Analyst Agent ─────────────────────────────────────────────────────────────

ANALYST_PROMPT = ChatPromptTemplate.from_template("""You are an analytical reasoning expert.
Analyse the provided information deeply and critically.

Structure your response with:
- **Analysis**: Core findings and patterns
- **Implications**: What this means
- **Pros & Cons** or **Strengths & Weaknesses** (where applicable)
- **Evidence**: Specific quotes/data from documents
- **Conclusion**: Your reasoned judgment

Document context:
{context}

Analysis request: {question}

Provide deep analytical insight:""")


class AnalystAgent:
    name = "analyst_agent"
    description = "Performs analytical reasoning and evaluations"

    def __init__(self, model: str, base_url: str, retriever: Any | None = None):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.3)
        self.retriever = retriever

    def set_retriever(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> tuple[str, list[dict]]:
        if not self.retriever:
            return "No documents available for analysis.", []

        docs = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = ANALYST_PROMPT | self.llm
        response = chain.invoke({"context": context, "question": query})
        sources = [{"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")} for d in docs[:3]]
        return response.content, sources

    def stream(self, query: str):
        if not self.retriever:
            yield "No documents available.", []
            return

        docs = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        chain = ANALYST_PROMPT | self.llm
        sources = [{"file": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?")} for d in docs[:3]]

        for chunk in chain.stream({"context": context, "question": query}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token, None
        yield "", sources


# ── General Agent ─────────────────────────────────────────────────────────────

GENERAL_PROMPT = ChatPromptTemplate.from_template("""You are a knowledgeable general assistant.
Answer the following question using your training knowledge.
Be clear, accurate, and helpful.

Question: {question}

Answer:""")


class GeneralAgent:
    name = "general_agent"
    description = "Answers general knowledge questions"

    def __init__(self, model: str, base_url: str, **kwargs):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.4)

    def set_retriever(self, retriever):
        pass  # General agent doesn't use retriever

    def run(self, query: str) -> tuple[str, list[dict]]:
        chain = GENERAL_PROMPT | self.llm
        response = chain.invoke({"question": query})
        return response.content, []

    def stream(self, query: str):
        chain = GENERAL_PROMPT | self.llm
        for chunk in chain.stream({"question": query}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token, None
        yield "", []
