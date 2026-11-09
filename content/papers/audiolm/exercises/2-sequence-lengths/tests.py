from submission import sequence_lengths

WANT_10 = {
    "semantic": 250,
    "acoustic_coarse": 2000,
    "acoustic_all": 6000,
    "stage1": 250,
    "stage2": 2250,
    "stage3_chunk": 1800,
    "stage3_chunks": 4,
    "one_stage": 6250,
}


def test_ten_second_clip():
    got = sequence_lengths(10)
    for k, v in WANT_10.items():
        assert got.get(k) == v, f"{k}: expected {v}, got {got.get(k)}"


def test_semantic_is_half_the_acoustic_frame_count():
    got = sequence_lengths(6)
    assert got["acoustic_all"] == 12 * 2 * got["semantic"], f"expected acoustic_all = 12 * 2 * semantic = {24 * got['semantic']}, got {got['acoustic_all']}"


def test_stage2_has_only_coarse_layers():
    got = sequence_lengths(4)
    assert got["stage2"] == got["semantic"] + got["acoustic_coarse"], f"expected {got['semantic'] + got['acoustic_coarse']}, got {got['stage2']}"
    assert got["acoustic_coarse"] == 200 * 4, f"4 s at 50 Hz and Q'=4 is 800 tokens, got {got['acoustic_coarse']}"


def test_stage3_length_is_independent_of_clip_length():
    a = sequence_lengths(3)["stage3_chunk"]
    b = sequence_lengths(120)["stage3_chunk"]
    assert a == b == 1800, f"expected 1800 for both, got {a} and {b}"


def test_chunk_count_rounds_up():
    assert sequence_lengths(3)["stage3_chunks"] == 1, "3 s is one 3 s chunk"
    assert sequence_lengths(3.5)["stage3_chunks"] == 2, f"3.5 s needs 2 chunks, got {sequence_lengths(3.5)['stage3_chunks']}"
    assert sequence_lengths(10)["stage3_chunks"] == 4, f"10 s needs 4 chunks, got {sequence_lengths(10)['stage3_chunks']}"


def test_other_configuration():
    got = sequence_lengths(2, Q=8, Qp=2, chunk_s=1)
    want = {"semantic": 50, "acoustic_coarse": 200, "acoustic_all": 800, "stage2": 250, "stage3_chunk": 400, "stage3_chunks": 2, "one_stage": 850}
    for k, v in want.items():
        assert got[k] == v, f"{k}: expected {v}, got {got[k]}"
