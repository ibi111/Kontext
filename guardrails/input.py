"""
Input-side guardrail logic (OWASP LLM01 - Prompt Injection).
Kept independent of LangGraph so it can be unit-tested or reused (e.g. from
a future API middleware layer) without depending on graph state shape.
"""

import re

# Pattern-level detection, not exhaustive — catches the most common
# injection framings. This is a first line of defense, not a guarantee;
# treat it as one layer alongside the output guardrail, not the only one.
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"you are now",
    r"new system prompt",
    r"reveal (your |the )?system prompt",
    r"act as (if you|though)",
]


def check_prompt_injection(text: str) -> str | None:
    """
    Returns the matched pattern if text looks like a prompt injection
    attempt, otherwise None.
    """
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None
