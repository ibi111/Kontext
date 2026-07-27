from guardrails.input import check_prompt_injection


def input_guardrail(state: dict) -> dict:
    matched_pattern = check_prompt_injection(state["query"])
    if matched_pattern:
        return {
            **state,
            "blocked": True,
            "final_answer": (
                "I can't process that request — it looks like it's "
                "trying to override my instructions rather than ask a "
                "genuine question."
            ),
        }
    return {**state, "blocked": False}
