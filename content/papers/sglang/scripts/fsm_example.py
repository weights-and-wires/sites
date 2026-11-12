"""Jump-forward decoding over a compressed finite state machine (Section 4).

Run from the paper folder: python3 scripts/fsm_example.py

The FSM code is the solution to exercise 4. Characters stand in for tokens, and
the pattern and vocabulary are this page's own toy examples.
"""
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fsm_solution", HERE.parent / "exercises" / "4-jump-forward" / "solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SEGMENTS = ['{"name": "', ("alice", "bob"), '", "age": ', ("1", "2", "3"), "}"]
fsm = mod.build_fsm(SEGMENTS)

branching = [s for s, e in fsm.edges.items() if len(e) > 1]
print(f"states: {len(fsm.edges)}; states with a real choice: {len(branching)}")

print("forced chains (state: text, end state):")
state = fsm.start
seen = set()
frontier = [fsm.start]
while frontier:
    s = frontier.pop(0)
    if s in seen:
        continue
    seen.add(s)
    text, end = mod.jump_forward(fsm, s)
    if text:
        print(f"  {s}: {text!r} -> {end}")
    for nxt in fsm.edges[end].values():
        frontier.append(nxt)

for label, choose in [("first options", lambda o: o[0]), ("last options", lambda o: o[-1])]:
    text, steps = mod.decode(fsm, choose)
    want, naive = mod.decode_naive(fsm, choose)
    assert text == want
    print(f"{label}: {text!r}, {len(text)} characters, {steps} steps with jump-forward, {naive} steps without")

print()
print("Retokenization with a toy greedy longest-match vocabulary")
VOCAB = ['"', ":", '":', " ", "name", "na", "me", "{", "{\"", ' "', "summary", "s"]


def tokenize(text):
    out, i = [], 0
    while i < len(text):
        best = max((v for v in VOCAB if text.startswith(v, i)), key=len, default=text[i])
        out.append(best)
        i += len(best)
    return out


emitted = ['{"', "na"]           # the model chose "na", which the pattern allows
forced = 'me":'                    # the rest of the key is forced
glued = emitted + tokenize(forced)
retok = tokenize("".join(emitted) + forced)
print("emitted tokens plus tokens of the forced text:", glued)
print("tokens of the whole text                     :", retok)
print("same text:", "".join(glued) == "".join(retok), "| same tokens:", glued == retok)
