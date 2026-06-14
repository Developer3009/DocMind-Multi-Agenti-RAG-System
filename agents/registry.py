"""
Agent Registry — builds and caches all agents for a given model config.
"""

from __future__ import annotations
from agents.orchestrator import OrchestratorAgent
from agents.rag_agent import RAGAgent
from agents.specialist_agents import SummaryAgent, ComparisonAgent, AnalystAgent, GeneralAgent

AGENT_CLASSES = {
    "rag_agent":        RAGAgent,
    "summary_agent":    SummaryAgent,
    "comparison_agent": ComparisonAgent,
    "analyst_agent":    AnalystAgent,
    "general_agent":    GeneralAgent,
}

AGENT_ICONS = {
    "rag_agent":        "🔍",
    "summary_agent":    "📋",
    "comparison_agent": "⚖️",
    "analyst_agent":    "🧠",
    "general_agent":    "💡",
    "orchestrator":     "🎯",
}

AGENT_DESCRIPTIONS = {
    "rag_agent":        "Document QA — finds precise answers from your PDFs",
    "summary_agent":    "Summariser — creates structured summaries & outlines",
    "comparison_agent": "Comparator — tables and side-by-side analysis",
    "analyst_agent":    "Analyst — deep reasoning, pros/cons, implications",
    "general_agent":    "General — knowledge questions outside your documents",
}


class AgentRegistry:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        self.orchestrator = OrchestratorAgent(model=model, base_url=base_url)
        self._agents: dict = {}
        self._build_agents()

    def _build_agents(self):
        for name, cls in AGENT_CLASSES.items():
            self._agents[name] = cls(model=self.model, base_url=self.base_url)

    def set_retriever(self, retriever):
        for agent in self._agents.values():
            agent.set_retriever(retriever)

    def get(self, name: str):
        return self._agents.get(name)

    def route(self, query: str) -> dict:
        return self.orchestrator.route(query)

    def run_agent(self, agent_name: str, query: str) -> tuple[str, list]:
        agent = self.get(agent_name)
        if agent:
            return agent.run(query)
        return f"Unknown agent: {agent_name}", []

    def stream_agent(self, agent_name: str, query: str):
        agent = self.get(agent_name)
        if agent:
            yield from agent.stream(query)
        else:
            yield f"Unknown agent: {agent_name}", []

    def synthesise(self, query: str, outputs: dict[str, str]) -> str:
        return self.orchestrator.synthesise(query, outputs)
