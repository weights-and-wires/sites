from submission import FSM, build_fsm, decode, decode_naive, jump_forward

SEGMENTS = ['{"name": "', ("alice", "bob"), '", "age": ', ("1", "2", "3"), "}"]


def pick_last(options):
    return options[-1]


def pick_first(options):
    return options[0]


def test_jump_forward_follows_a_literal():
    fsm = build_fsm(['{"a": ', ("x", "y")])
    text, state = jump_forward(fsm, fsm.start)
    assert text == '{"a": ', f"expected the whole literal, got {text!r}"
    assert len(fsm.edges[state]) == 2, f"it stops at the choice between x and y, got edges {sorted(fsm.edges[state])}"


def test_jump_forward_does_nothing_at_a_branch():
    fsm = build_fsm([("x", "y")])
    text, state = jump_forward(fsm, fsm.start)
    assert (text, state) == ("", fsm.start), f"a branching state has nothing forced, got {(text, state)}"


def test_jump_forward_stops_before_a_choice_inside_alternatives():
    fsm = build_fsm([("alice", "alan")])
    text, state = jump_forward(fsm, fsm.start)
    assert text == "al", f"'a' and 'l' are shared, then 'i' or 'a' is a choice; expected 'al', got {text!r}"
    assert sorted(fsm.edges[state]) == ["a", "i"], f"expected the choice a or i, got {sorted(fsm.edges[state])}"


def test_jump_forward_stops_at_an_accepting_state():
    fsm = FSM({0: {"a": 1}, 1: {"b": 2}, 2: {}}, 0, {1, 2})
    text, state = jump_forward(fsm, 0)
    assert (text, state) == ("a", 1), f"state 1 is accepting, so the model may stop there; expected ('a', 1), got {(text, state)}"


def test_jump_forward_at_the_end():
    fsm = build_fsm(["ab"])
    text, state = jump_forward(fsm, fsm.start)
    assert text == "ab", f"expected 'ab', got {text!r}"
    assert fsm.edges[state] == {}, "the end state has no outgoing edges"
    assert jump_forward(fsm, state) == ("", state), "nothing is forced at the end"


def test_decode_counts_only_choices():
    fsm = build_fsm(SEGMENTS)
    text, steps = decode(fsm, pick_last)
    assert text == '{"name": "bob", "age": 3}', f"expected the bob/3 string, got {text!r}"
    assert steps == 2, f"two choices (alice or bob, then a digit): expected 2 steps, got {steps}"


def test_decode_matches_naive_decoding():
    fsm = build_fsm(SEGMENTS)
    for choose in (pick_first, pick_last):
        text, steps = decode(fsm, choose)
        want, naive_steps = decode_naive(fsm, choose)
        assert text == want, f"jump-forward must not change the output: expected {want!r}, got {text!r}"
        assert naive_steps == len(want), f"naive decoding takes one step per character: expected {len(want)}, got {naive_steps}"
        assert steps < naive_steps, f"jump-forward must take fewer steps than {naive_steps}, got {steps}"


def test_decode_passes_the_options_in_order():
    fsm = build_fsm([("b", "a", "c")])
    seen = []

    def choose(options):
        seen.append(list(options))
        return options[0]

    text, steps = decode(fsm, choose)
    assert seen == [["a", "b", "c"]], f"options are sorted characters; expected [['a', 'b', 'c']], got {seen}"
    assert (text, steps) == ("a", 1), f"expected ('a', 1), got {(text, steps)}"


def test_decode_offers_stop_at_an_accepting_state():
    fsm = FSM({0: {"a": 1}, 1: {"b": 2}, 2: {}}, 0, {1, 2})
    seen = []

    def stop(options):
        seen.append(list(options))
        return None

    text, steps = decode(fsm, stop)
    assert seen == [["b", None]], f"at the accepting state the options are ['b', None], got {seen}"
    assert (text, steps) == ("a", 1), f"stopping costs the step that chose it: expected ('a', 1), got {(text, steps)}"
    text, steps = decode(fsm, pick_first)
    assert (text, steps) == ("ab", 1), f"choosing 'b' ends at a state with no edges: expected ('ab', 1), got {(text, steps)}"


def test_decode_collects_text_before_the_first_choice():
    fsm = build_fsm(["hi ", ("x", "y")])
    text, steps = decode(fsm, pick_first)
    assert text == "hi x", f"the forced prefix belongs in the output, expected 'hi x', got {text!r}"
    assert steps == 1, f"expected 1 step, got {steps}"
