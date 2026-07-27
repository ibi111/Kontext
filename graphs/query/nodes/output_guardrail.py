from guardrails.output import redact_pii


def output_guardrail(state: dict) -> dict:
    return {**state, "final_answer": redact_pii(state["final_answer"])}
