from submission import receptive_field, measured_receptive_field

BLOCK = [2**i for i in range(10)]

CASES = [
    ("figure 2: four undilated layers", [1, 1, 1, 1], 2, 5),
    ("figure 3: dilations 1, 2, 4, 8", [1, 2, 4, 8], 2, 16),
    ("one block 1..512", BLOCK, 2, 1024),
    ("three blocks", BLOCK * 3, 2, 3070),
    ("filter length 3", BLOCK, 3, 2047),
    ("single layer", [4], 2, 5),
    ("no layers", [], 2, 1),
]


def test_formula_values():
    for name, dil, k, want in CASES:
        got = receptive_field(dil, k)
        assert got == want, f"{name}: expected {want}, got {got}"


def test_measured_values():
    for name, dil, k, want in CASES:
        got = measured_receptive_field(dil, k)
        assert got == want, f"{name}: expected {want}, got {got}"


def test_formula_matches_measurement():
    for dil in [[1, 3, 9], [5, 1, 2, 2], [1, 2, 4, 1, 2, 4], [7]]:
        for k in (2, 3, 4):
            a, b = receptive_field(dil, k), measured_receptive_field(dil, k)
            assert a == b, f"dilations {dil}, k={k}: formula gave {a}, brute force gave {b}"


def test_returns_int():
    assert isinstance(receptive_field([1, 2], 2), int), f"expected int, got {type(receptive_field([1, 2], 2))}"
    assert isinstance(measured_receptive_field([1, 2], 2), int), "measured_receptive_field must return an int"
