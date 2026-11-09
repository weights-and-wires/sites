import re

import numpy as np
from submission import evaluate, format_prompt

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def make(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = rng.integers(3, 20, size=2)
        out.append((f"What is {a} blip {b}?", str(a * b)))
    return out


POOL = make(30, 1)
TEST = make(40, 2)


def learner(prompt):
    """Learns what 'blip' means from the demonstrations; answers 0 with none."""
    pairs = re.findall(r"Q: What is (\d+) blip (\d+)\?\nA: (-?\d+)\n", prompt)
    a, b = map(int, re.findall(r"Q: What is (\d+) blip (\d+)\?\nA:$", prompt)[0])
    for f in OPS.values():
        if pairs and all(f(int(x), int(y)) == int(z) for x, y, z in pairs):
            return " " + str(f(a, b))
    return " 0"


def test_prompt_format():
    got = format_prompt([("What is 1 plus 2?", "3")], "What is 4 plus 5?")
    want = "Q: What is 1 plus 2?\nA: 3\n\nQ: What is 4 plus 5?\nA:"
    assert got == want, f"expected {want!r}, got {got!r}"


def test_prompt_without_demos():
    got = format_prompt([], "What is 4 plus 5?")
    assert got == "Q: What is 4 plus 5?\nA:", f"expected 'Q: What is 4 plus 5?\\nA:', got {got!r}"


def test_prompt_order_and_description():
    demos = [("q1", "a1"), ("q2", "a2")]
    got = format_prompt(demos, "q3", description="Add the numbers.")
    want = "Add the numbers.\n\nQ: q1\nA: a1\n\nQ: q2\nA: a2\n\nQ: q3\nA:"
    assert got == want, f"expected {want!r}, got {got!r}"


def test_zero_shot_cannot_learn_the_operation():
    acc = evaluate(learner, TEST, POOL, k=0)
    assert acc == 0.0, f"with no demonstrations the stub must fail every query; expected 0.0, got {acc}"


def test_one_shot_is_enough_for_the_learner():
    acc = evaluate(learner, TEST, POOL, k=1)
    assert acc == 1.0, f"the stub infers 'blip' from one demonstration; expected 1.0, got {acc}"
    acc5 = evaluate(learner, TEST, POOL, k=5)
    assert acc5 == 1.0, f"expected 1.0 at k=5, got {acc5}"


def test_completion_is_stripped():
    oracle = lambda p: " " + str(int(re.findall(r"(\d+) blip (\d+)\?\nA:$", p)[0][0]) * int(re.findall(r"(\d+) blip (\d+)\?\nA:$", p)[0][1])) + "\n"
    acc = evaluate(oracle, TEST, POOL, k=1)
    assert acc == 1.0, f"a completion of ' 42\\n' should match '42' after stripping; expected 1.0, got {acc}"


def test_match_is_exact_not_substring():
    def long_answer(p):
        a, b = map(int, re.findall(r"(\d+) blip (\d+)\?\nA:$", p)[0])
        return f" {a * b}5"

    acc = evaluate(long_answer, TEST, POOL, k=1)
    assert acc == 0.0, f"'1245' is not '124'; expected 0.0, got {acc}"


def test_partial_accuracy():
    def right_on_even_first_operand(p):
        a, b = map(int, re.findall(r"(\d+) blip (\d+)\?\nA:$", p)[0])
        return " " + str(a * b if a % 2 == 0 else -1)

    want = np.mean([int(re.findall(r"\d+", q)[0]) % 2 == 0 for q, _ in TEST])
    acc = evaluate(right_on_even_first_operand, TEST, POOL, k=1)
    assert abs(acc - want) < 1e-12, f"expected accuracy {want}, got {acc}"


def test_demos_never_contain_the_query():
    prompts = []

    def record(p):
        prompts.append(p)
        return ""

    evaluate(record, POOL, POOL, k=5)
    assert len(prompts) == len(POOL), f"expected {len(POOL)} calls, got {len(prompts)}"
    for p in prompts:
        final = re.findall(r"Q: (.*)\nA:$", p)[0]
        assert p.count(f"Q: {final}\n") == 1, f"the query {final!r} also appears as a demonstration"


def test_demos_are_distinct_and_counted():
    prompts = []

    def record(p):
        prompts.append(p)
        return ""

    small_pool = POOL[:7]
    evaluate(record, TEST[:10], small_pool, k=5)
    for p in prompts:
        qs = re.findall(r"^Q: (.*)$", p, flags=re.M)
        assert len(qs) == 6, f"expected 5 demonstrations plus the query, found {len(qs)} questions"
        assert len(set(qs[:-1])) == 5, f"demonstrations must be drawn without replacement, got {qs[:-1]}"


def test_seed_controls_the_draw():
    def run(seed):
        prompts = []
        evaluate(lambda p: prompts.append(p) or "", TEST[:5], POOL, k=3, seed=seed)
        return prompts

    assert run(0) == run(0), "the same seed must give the same prompts"
    assert run(0) != run(1), "different seeds should give different demonstrations"


def test_too_few_demonstrations():
    try:
        evaluate(learner, TEST[:2], POOL[:3], k=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when k exceeds the number of available demonstrations")


def test_one_generator_is_shared_across_queries():
    prompts = []
    evaluate(lambda p: prompts.append(p) or "", TEST[:6], POOL, k=3)
    demo_sets = {tuple(re.findall(r"^Q: (.*)$", p, flags=re.M)[:-1]) for p in prompts}
    assert len(demo_sets) > 1, "every query got the same demonstrations; create the generator once, outside the loop"
