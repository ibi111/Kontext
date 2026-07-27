"""
Output-side guardrail logic (OWASP LLM06 - Sensitive Information
Disclosure). Kept independent of LangGraph so it can be unit-tested or
reused elsewhere without depending on graph state shape.
"""

import re

# Pattern-level PII redaction — SSN-like, credit-card-like, email. Not
# exhaustive, but catches the highest-risk categories for a document-QA
# system whose corpus may include real corporate/personal data.
PII_PATTERNS = {
    "SSN-like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card-like": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED-{label}]", text)
    return text
