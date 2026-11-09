from submission import unet_output_size


def test_paper_size():
    got = unet_output_size(572)
    assert got == 388, f"expected 388 for a 572 input (Figure 1), got {got}"


def test_other_valid_size():
    got = unet_output_size(588)
    assert got == 404, f"expected 404 for a 588 input, got {got}"


def test_odd_pool_is_invalid():
    got = unet_output_size(573)
    assert got is None, f"expected None for 573, because the first pool sees 569 (odd), got {got}"


def test_odd_size_deeper_in_the_network():
    # The first pool sees 570 (even) and returns 285, so the second pool sees 281 (odd).
    got = unet_output_size(574)
    assert got is None, f"expected None for 574, because the second pool sees 281 (odd), got {got}"


def test_too_small():
    got = unet_output_size(60)
    assert got is None, f"expected None for 60, because the bottom collapses, got {got}"


def test_one_level():
    got = unet_output_size(20, levels=1)
    assert got == 4, f"expected 4 for n=20, levels=1 (20 -> 16 -> 8 -> 4 -> 8 -> 4), got {got}"


def test_bottom_collapse_in_one_level():
    got = unet_output_size(8, levels=1)
    assert got is None, f"expected None for n=8, levels=1 (the bottom size is 0), got {got}"


def test_zero_levels_is_two_convs():
    got = unet_output_size(10, levels=0)
    assert got == 6, f"expected 6 for levels=0 (one bottom pair, minus 4), got {got}"


def test_output_is_smaller_than_input():
    for n in range(100, 700):
        out = unet_output_size(n)
        if out is not None:
            assert out < n, f"output {out} must be smaller than input {n}"
