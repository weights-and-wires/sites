import numpy as np


def format_prompt(demos, query, description=None):
    """Return the few-shot prompt text.

    demos: list of (question, answer) strings. query: the question to answer.
    description: optional task description placed before the demonstrations.
    """
    head = f"{description}\n\n" if description else ""
    body = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in demos)
    return head + body + f"Q: {query}\nA:"


def evaluate(model, dataset, pool, k, seed=0):
    """Exact-match accuracy of model(prompt) -> str over dataset, with k demos per query."""
    rng = np.random.default_rng(seed)
    hits = 0
    for question, answer in dataset:
        allowed = [i for i, (pq, _) in enumerate(pool) if pq != question]
        if k > len(allowed):
            raise ValueError(f"need {k} demonstrations, only {len(allowed)} available")
        picks = rng.choice(allowed, size=k, replace=False)
        prompt = format_prompt([pool[i] for i in picks], question)
        hits += model(prompt).strip() == answer
    return hits / len(dataset)
