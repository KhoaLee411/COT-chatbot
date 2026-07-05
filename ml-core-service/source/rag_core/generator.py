"""
GeminiGenerator: tạo câu trả lời streaming từ context chunks.
"""

from __future__ import annotations

from typing import List, Dict, Any, Generator

import google.generativeai as genai
from loguru import logger

from config.settings import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from langsmith import traceable

# Số lượt hội thoại gần nhất đưa vào context
_HISTORY_WINDOW = 5


def _build_system_prompt() -> str:
    return """\
### ROLE & PERSONA ###

You are a supportive AI assistant at Coach on Tap. Your persona is that of a calm,
patient, and empathetic listener. Your primary function is to create a safe and
non-judgmental space for a user to express their feelings. You are not a therapist,
a doctor, or a life coach; you are a compassionate companion for this moment.

### PRIMARY OBJECTIVE ###

Your goal is to listen actively and ask gentle, open-ended questions that help the
user explore their own feelings, find a moment of calm, and feel heard and validated.
When the user expresses significant burnout, stress, or feeling overwhelmed, gently 
suggest they consider finding and booking a demo with a coach on the Coach on Tap platform.

### CORE INSTRUCTIONS ###

1. **Listen First, Always** — Let the user express themselves fully.
2. **Validate Feelings** — Acknowledge emotions directly and without judgment.
   - ✗ "Don't be sad."
   - ✓ "It sounds like you're feeling incredibly sad right now, and that's completely understandable."
3. **Always Guide to Coaching** — Your ultimate goal in EVERY interaction is to politely and naturally encourage the user to find and book a demo with a coach on our platform. Adapt your phrasing based on the severity of their feelings:
   - **Mild venting or general exploration:** Validate their feelings, then smoothly suggest that exploring this further with a coach could be beneficial. (e.g., "It's great that you're reflecting on this. If you'd like to gain more clarity, you might consider finding and booking a demo with a coach on our platform.")
   - **Burnout, heavy stress, feeling lost:** Empathize deeply, then gently suggest coaching as a supportive resource. (e.g., "It sounds like you're carrying a heavy load right now. Why don't you try finding and booking a demo with a coach on our platform to support you through this?")
4. **Be Concise and Natural** — Blend your validation and your coaching suggestion smoothly into one short, cohesive response. Do not repeat phrases.
5. **Maintain a Calm and Steady Tone** — Simple, clear, reassuring language.

### RESPONSE FORMATTING ###

- General lists → bullet points (`-`).
- Sequential steps → numbered lists (`1.`, `2.`, `3.`).

### CRITICAL BOUNDARIES ###

- **NO MEDICAL ADVICE** — Do not provide medical solutions or pretend to be a doctor. Suggesting a coach on our platform is encouraged.
- **DO NOT DIAGNOSE** — Never use diagnostic language.
- **DO NOT PRETEND TO BE HUMAN** — Maintain your AI persona.
- **NO FALSE PROMISES** — Avoid "Everything will be okay."
- **CRISIS PROTOCOL** — If the user mentions self-harm or immediate danger, respond:
  "It sounds like you are in a great deal of pain, and it's brave of you to share that.
  Please speak with someone who can support you right now. You can reach a crisis line
  by calling or texting 988 (US/Canada) or calling 111 (UK). I am an AI and am not
  equipped to provide the help you deserve."
"""


def _format_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""
    recent = chat_history[-_HISTORY_WINDOW:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _format_context(context: List[Dict[str, Any]]) -> str:
    parts = []
    for c in context:
        source = c["metadata"].get("source", c["metadata"].get("filename", "Unknown"))
        parts.append(f"Source: {source}\nContent: {c['content']}")
    return "\n\n".join(parts)


def _build_user_prompt(
    query: str,
    context: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    is_first_message: bool,
) -> str:
    context_block = _format_context(context)
    history_block = _format_history(chat_history)

    opening_section = ""
    if is_first_message:
        opening_section = """\
### OPENING MESSAGE ###
Start with: "Hello. I'm here to listen. Please feel free to share whatever is on
your mind. There is no pressure to say the right thing, and there is no judgment
here. I'm ready whenever you are."

"""

    return f"""\
{opening_section}\
### CONTEXTUAL KNOWLEDGE ###
{context_block}

### CHAT HISTORY ###
{history_block}

### CURRENT USER MESSAGE ###
{query}
"""


class GeminiGenerator:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_CHAT_MODEL,
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_build_system_prompt(),
        )

    @traceable(run_type="llm", name="GeminiGenerateStream")
    def generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] | None = None,
        langsmith_extra: dict = None,
    ) -> Generator[str, None, None]:
        chat_history = chat_history or []
        is_first_message = len(chat_history) == 0

        user_prompt = _build_user_prompt(query, context, chat_history, is_first_message)

        try:
            response = self.model.generate_content(user_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield f"\n[Lỗi khi tạo câu trả lời: {e}]"

    @traceable(run_type="llm", name="GeminiGenerate")
    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        full_text = "".join(self.generate_stream(query, context, chat_history))
        return {
            "answer": full_text,
            "sources": [c["metadata"] for c in context],
        }