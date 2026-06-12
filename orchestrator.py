"""
Orchestrator Agent
Decides which specialist agent(s) should handle a user query,
then synthesises their outputs into a final response.
"""

from __future__ import annotations
import json
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


ROUTER_SYSTEM = """You are an intelligent routing agent. Given a user question and a list of
available specialist agents, decide which agent(s) should handle it.

Agents available:
- rag_agent: Answers questions from uploaded PDF documents. Use for document-specific questions.
- summary_agent: Summarises long documents or produces structured summaries/outlines.
- comparison_agent: Compares information across multiple documents or topics.
- analyst_agent: Performs analytical reasoning — trends, implications, pros/cons, evaluations.
- general_agent: Handles general knowledge questions not covered by documents.

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{
  "agents": ["agent_name", ...],
  "reason": "one-sentence explanation"
}
"""


class OrchestratorAgent:
    """Routes queries to specialist agents and synthesises results."""

    def __init__(self, model: str, base_url: str):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        self.model = model
        self.base_url = base_url

    def route(self, query: str) -> dict[str, Any]:
        """Return routing decision as a dict."""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=f"User query: {query}"),
            ])
            text = resp.content.strip()
            # Strip possible markdown fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception:
            return {"agents": ["rag_agent"], "reason": "Default routing due to parse error."}

    def synthesise(self, query: str, agent_outputs: dict[str, str]) -> str:
        """
        Combine outputs from multiple agents into a single coherent answer.
        Only called when >1 agent was used.
        """
        parts = "\n\n".join(
            f"[{agent.upper()}]\n{output}"
            for agent, output in agent_outputs.items()
        )
        prompt = f"""You received these outputs from specialist agents answering:
"{query}"

{parts}

Synthesise them into one clear, well-structured answer. Do not repeat yourself.
If agents contradict, note it. Be concise."""
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        return resp.content
