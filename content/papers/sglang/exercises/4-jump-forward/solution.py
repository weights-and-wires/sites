class FSM:
    """Finished for you. A character-level finite state machine."""

    def __init__(self, edges, start, accepting):
        self.edges = edges            # state -> {character: next state}
        self.start = start
        self.accepting = set(accepting)


def build_fsm(segments):
    """Finished for you. Build an FSM from a list of segments.

    A segment is a string (a literal) or a tuple of strings (one of these
    alternatives). No alternative may be a prefix of another one.
    """
    edges = {0: {}}

    def new_state():
        edges[len(edges)] = {}
        return len(edges) - 1

    cur = 0
    for seg in segments:
        alts = [seg] if isinstance(seg, str) else list(seg)
        join = new_state()
        for alt in alts:
            s = cur
            for j, ch in enumerate(alt):
                if j == len(alt) - 1:
                    if ch in edges[s]:
                        raise ValueError(f"alternative {alt!r} is a prefix of another or repeats one")
                    edges[s][ch] = join
                else:
                    if ch not in edges[s]:
                        edges[s][ch] = new_state()
                    s = edges[s][ch]
        cur = join
    return FSM(edges, 0, {cur})


def jump_forward(fsm, state):
    """Follow the forced chain from `state` and return (text, end_state).

    A state is forced when it has exactly one outgoing edge and is not
    accepting (an accepting state can also end the output, so the model has a
    real choice there). Follow forced states and collect their characters.
    Stop at the first state that is not forced.
    """
    text = []
    while len(fsm.edges[state]) == 1 and state not in fsm.accepting:
        (ch, nxt), = fsm.edges[state].items()
        text.append(ch)
        state = nxt
    return "".join(text), state


def decode(fsm, choose):
    """Generate one string with jump-forward decoding and return (text, steps).

    Loop: call jump_forward from the current state and append its text. If the
    new state has no outgoing edges, stop. Otherwise the model must choose: call
    `choose(options)` once, where `options` is the sorted list of allowed
    characters, followed by None when the state is accepting (None means stop).
    Each call to `choose` is one step (one model forward pass). If `choose`
    returns None, stop. Otherwise append the character and move along its edge.
    """
    state, out, steps = fsm.start, [], 0
    while True:
        text, state = jump_forward(fsm, state)
        out.append(text)
        edges = fsm.edges[state]
        if not edges:
            break
        options = sorted(edges) + ([None] if state in fsm.accepting else [])
        ch = choose(options)
        steps += 1
        if ch is None:
            break
        out.append(ch)
        state = edges[ch]
    return "".join(out), steps


def decode_naive(fsm, choose):
    """Finished for you. One step per character, as in per-token masking."""
    state, out, steps = fsm.start, [], 0
    while True:
        edges = fsm.edges[state]
        if not edges:
            break
        options = sorted(edges) + ([None] if state in fsm.accepting else [])
        ch = choose(options) if len(options) > 1 else options[0]
        steps += 1
        if ch is None:
            break
        out.append(ch)
        state = edges[ch]
    return "".join(out), steps
