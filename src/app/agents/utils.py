def normalize_answer(answer) -> str:
    """Normalize the diagnostic answer by turning it into a string."""
    if answer is None:
        return ""
    elif isinstance(answer, str):
        return answer
    else:
        return answer.model_dump_json()