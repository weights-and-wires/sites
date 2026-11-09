from submission import count_params

MIXTRAL = dict(dim=4096, n_layers=32, head_dim=128, hidden_dim=14336, n_heads=32,
               n_kv_heads=8, vocab_size=32000, num_experts=8, top_k_experts=2)

# A small config that can be counted by hand:
# attention = 2*8*(4*2) + 2*8*(2*2) = 192, expert = 3*8*16 = 384, router = 32, norms = 16
# layer total = 192 + 4*384 + 32 + 16 = 1776, layer active = 192 + 2*384 + 32 + 16 = 1008
# embeddings = 2*10*8 = 160, final gain = 8
# total = 2*1776 + 168 = 3720, active = 2*1008 + 168 = 2184
SMALL = dict(dim=8, n_layers=2, head_dim=2, hidden_dim=16, n_heads=4,
             n_kv_heads=2, vocab_size=10, num_experts=4, top_k_experts=2)


def test_small_config_by_hand():
    got = count_params(SMALL)
    assert got["total"] == 3720, f"expected total 3720, got {got['total']}"
    assert got["active"] == 2184, f"expected active 2184, got {got['active']}"


def test_mixtral_total():
    got = count_params(MIXTRAL)["total"]
    assert got == 46_702_792_704, f"expected total 46702792704, got {got}"


def test_mixtral_active():
    got = count_params(MIXTRAL)["active"]
    assert got == 12_879_925_248, f"expected active 12879925248, got {got}"


def test_values_are_integers():
    got = count_params(MIXTRAL)
    assert isinstance(got["total"], int) and isinstance(got["active"], int), f"expected ints, got {type(got['total'])} and {type(got['active'])}"


def test_dense_model_has_no_router_and_equal_counts():
    dense = dict(MIXTRAL, num_experts=1, top_k_experts=1)
    got = count_params(dense)
    assert got["total"] == got["active"], f"a dense model must have total == active, got {got}"
    assert got["total"] == 7_241_732_096, f"expected 7241732096 for the one-expert shape, got {got['total']}"


def test_top_k_equal_to_num_experts_activates_everything():
    full = dict(MIXTRAL, top_k_experts=8)
    got = count_params(full)
    assert got["active"] == got["total"], f"top_k == num_experts means active == total, got {got}"


def test_more_experts_grow_total_but_not_active():
    a = count_params(MIXTRAL)
    b = count_params(dict(MIXTRAL, num_experts=16))
    assert b["total"] > a["total"], f"16 experts must have a larger total, got {b['total']} vs {a['total']}"
    router_growth = 32 * 4096 * 8
    assert b["active"] - a["active"] == router_growth, f"active may grow only by the router, expected +{router_growth}, got +{b['active'] - a['active']}"
