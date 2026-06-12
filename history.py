"""
Chat history helper — wraps Streamlit session state.
"""

from __future__ import annotations
import streamlit as st
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    agent_used: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    routing_reason: str = ""


def init_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role: str, content: str, agent_used=None, sources=None, routing_reason=""):
    st.session_state.chat_history.append(ChatMessage(
        role=role,
        content=content,
        agent_used=agent_used or [],
        sources=sources or [],
        routing_reason=routing_reason,
    ))


def get_history() -> list[ChatMessage]:
    return st.session_state.get("chat_history", [])


def clear_history():
    st.session_state.chat_history = []
