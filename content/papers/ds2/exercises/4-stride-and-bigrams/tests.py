from submission import to_bigrams, output_length, min_ctc_frames, max_stride


def test_paper_example():
    got = to_bigrams("the cat sat")
    want = ["th", "e", " ", "ca", "t", " ", "sa", "t"]
    assert got == want, f"expected {want}, got {got}"


def test_bigrams_never_cross_word_boundaries():
    got = to_bigrams("ab cd")
    want = ["ab", " ", "cd"]
    assert got == want, f"expected {want}, got {got}"
    got = to_bigrams("a bc")
    want = ["a", " ", "bc"]
    assert got == want, f"expected {want}, got {got}"


def test_bigrams_round_trip():
    text = "it's a deep speech"
    got = "".join(to_bigrams(text))
    assert got == text, f"expected the tokens to join back into {text!r}, got {got!r}"


def test_odd_word_ends_in_a_unigram():
    got = to_bigrams("hello")
    want = ["he", "ll", "o"]
    assert got == want, f"expected {want}, got {got}"


def test_single_space_tokens_for_one_space():
    got = to_bigrams("a b")
    assert got == ["a", " ", "b"], f"expected ['a', ' ', 'b'], got {got}"


def test_output_length_uses_ceiling_at_every_layer():
    assert output_length(10, [2, 2]) == 3, f"expected ceil(ceil(10 / 2) / 2) = 3, got {output_length(10, [2, 2])}"
    assert output_length(9, [2, 2]) == 3, f"expected ceil(ceil(9 / 2) / 2) = 3, got {output_length(9, [2, 2])}"
    assert output_length(100, [2]) == 50, f"expected 50, got {output_length(100, [2])}"
    assert output_length(101, [2]) == 51, f"expected 51, got {output_length(101, [2])}"


def test_output_length_with_stride_one_and_no_layers():
    assert output_length(37, [1, 1, 1]) == 37, f"expected 37, got {output_length(37, [1, 1, 1])}"
    assert output_length(37, []) == 37, f"expected 37 for no layers, got {output_length(37, [])}"


def test_table_4_time_strides():
    # Three 2-D layers with strides 2x2, 2x1, 2x1 (frequency x time) shrink time by 2 overall.
    assert output_length(1001, [2, 1, 1]) == 501, f"expected 501, got {output_length(1001, [2, 1, 1])}"


def test_min_ctc_frames_counts_tokens_and_repeats():
    assert min_ctc_frames(list("hello")) == 6, f"expected 6 for h e l l o, got {min_ctc_frames(list('hello'))}"
    assert min_ctc_frames(["he", "ll", "o"]) == 3, f"expected 3, got {min_ctc_frames(['he', 'll', 'o'])}"
    assert min_ctc_frames(["aa", "aa"]) == 3, f"expected 3 for two equal tokens, got {min_ctc_frames(['aa', 'aa'])}"
    assert min_ctc_frames(["a", "a", "a"]) == 5, f"expected 5 for a a a, got {min_ctc_frames(['a', 'a', 'a'])}"
    assert min_ctc_frames([]) == 0, f"expected 0 for no tokens, got {min_ctc_frames([])}"


def test_max_stride_for_unigrams_and_bigrams():
    text = "the cat sat"
    uni = list(text)  # 11 tokens, no equal neighbors
    bi = to_bigrams(text)  # 8 tokens
    assert max_stride(100, uni) == 9, f"expected 9 for 11 tokens in 100 frames (ceil(100 / 9) = 12), got {max_stride(100, uni)}"
    assert max_stride(100, bi) == 14, f"expected 14 for 8 tokens in 100 frames (ceil(100 / 14) = 8), got {max_stride(100, bi)}"


def test_max_stride_with_repeats_and_impossible_cases():
    assert max_stride(6, list("hello")) == 1, f"expected 1 (6 frames needed, 6 available), got {max_stride(6, list('hello'))}"
    assert max_stride(5, list("hello")) == 0, f"expected 0 (6 frames needed, 5 available), got {max_stride(5, list('hello'))}"
