def run_program(src, x):
    """Run `def f(x)` from a source string and return f(x)."""
    env = {}
    exec(src, {"__builtins__": {"range": range, "sum": sum, "len": len, "abs": abs, "min": min, "max": max}}, env)
    return env["f"](x)


def check_answer(kind, program_src, task_input, task_output, answer):
    """Return True if the solver's answer is correct for this task kind.

    kind: "deduction", "abduction", or "induction".
    program_src: source of the task program (used by abduction).
    task_input: the input i (deduction, abduction) or a list of (x, y) pairs (induction).
    task_output: the output o (deduction, abduction).
    answer: predicted output, a proposed input, or a program source string.
    """
    raise NotImplementedError
