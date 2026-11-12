#!/usr/bin/env python3
"""The instruction format of Equation 1 (prefix on queries only).

Run from the paper folder: python3 scripts/query_format.py
"""

task = "Given a web search query, retrieve relevant passages that answer the query"
query = "how much protein should a female eat"

q_inst = f"Instruct: {task}\nQuery: {query}"
print(repr(q_inst))
print("characters added to the query:", len(q_inst) - len(query))
