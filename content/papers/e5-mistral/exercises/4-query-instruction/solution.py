def format_query(task, query):
    """Return 'Instruct: {task}\\nQuery: {query}' (Equation 1)."""
    return f"Instruct: {task}\nQuery: {query}"


def build_inputs(task, queries, docs):
    """Return (query_texts, doc_texts); only the queries get the instruction."""
    return [format_query(task, q) for q in queries], list(docs)


def add_eos(token_ids, eos_id):
    """Return a new list with eos_id appended. Don't modify token_ids."""
    return list(token_ids) + [eos_id]
