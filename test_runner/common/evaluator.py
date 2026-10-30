import operator

OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt
}


def evaluate_threshold(actual: float, op_str: str, threshold: float) -> bool:
    if op_str not in OPERATORS:
        raise ValueError(f"Unknown operator: {op_str}")

    op_func = OPERATORS[op_str]
    return op_func(actual, threshold)