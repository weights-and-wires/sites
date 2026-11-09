def cache_mha(n_h, d_h, layers):
    """Elements cached per token by multi-head attention: 2 n_h d_h l."""
    raise NotImplementedError


def cache_gqa(n_g, d_h, layers):
    """Elements cached per token by GQA with n_g groups: 2 n_g d_h l."""
    raise NotImplementedError


def cache_mla(d_c, d_hR, layers):
    """Elements cached per token by MLA: (d_c + d_hR) l."""
    raise NotImplementedError


def equivalent_gqa_groups(d_c, d_hR, d_h):
    """The (possibly fractional) group count where GQA matches MLA."""
    raise NotImplementedError
