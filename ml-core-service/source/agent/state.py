from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    query: str
    standalone_query: Optional[str]
    chat_history: List[Dict[str, str]]
    chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    is_out_of_scope: bool
    is_malicious: bool
    is_nonsense: bool
    is_greeting: bool
    error: Optional[str]