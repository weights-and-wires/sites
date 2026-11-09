from submission import allreduce_elements_per_layer, ring_traffic_per_gpu, logits_elements, padded_vocab


def test_small_layer_count():
    got = allreduce_elements_per_layer(1, 2, 4)
    assert got == 32, f"expected 4 * 1 * 2 * 4 = 32, got {got}"


def test_paper_8b_layer():
    got = allreduce_elements_per_layer(8, 1024, 3072)
    assert got == 100_663_296, f"expected 100663296, got {got}"


def test_count_scales_with_each_factor():
    base = allreduce_elements_per_layer(2, 3, 5)
    assert base == 120, f"expected 120, got {base}"
    assert allreduce_elements_per_layer(4, 3, 5) == 2 * base, "doubling the batch must double the volume"
    assert allreduce_elements_per_layer(2, 3, 10) == 2 * base, "doubling the hidden size must double the volume"


def test_ring_traffic():
    assert ring_traffic_per_gpu(800, 1) == 0, f"expected 0 with one GPU, got {ring_traffic_per_gpu(800, 1)}"
    assert ring_traffic_per_gpu(800, 2) == 800, f"expected 800 with 2 GPUs, got {ring_traffic_per_gpu(800, 2)}"
    assert ring_traffic_per_gpu(800, 8) == 1400, f"expected 1400 with 8 GPUs, got {ring_traffic_per_gpu(800, 8)}"
    got = ring_traffic_per_gpu(100_663_296, 8)
    assert got == 176_160_768, f"expected 176160768, got {got}"


def test_logits():
    assert logits_elements(8, 1024, 51200, False) == 419_430_400, f"expected 419430400, got {logits_elements(8, 1024, 51200, False)}"
    assert logits_elements(8, 1024, 51200, True) == 8192, f"expected 8192, got {logits_elements(8, 1024, 51200, True)}"


def test_fusion_ratio_is_vocab_size():
    ratio = logits_elements(3, 7, 1000, False) // logits_elements(3, 7, 1000, True)
    assert ratio == 1000, f"expected a ratio of 1000, got {ratio}"


def test_padded_vocab():
    assert padded_vocab(50257, 8) == 51200, f"expected 51200, got {padded_vocab(50257, 8)}"
    assert padded_vocab(50257, 1) == 50304, f"expected 50304, got {padded_vocab(50257, 1)}"
    assert padded_vocab(51200, 8) == 51200, f"an exact multiple must not grow; expected 51200, got {padded_vocab(51200, 8)}"
    assert padded_vocab(1025, 8) == 2048, f"expected 2048, got {padded_vocab(1025, 8)}"
