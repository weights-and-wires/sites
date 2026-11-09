#!/usr/bin/env python3
"""Parameter counts for the ViT variants in Table 1 (Section 4.1)."""


def vit_params(layers, d, mlp, p, c=3, n_tokens=None, image=224):
    n = (image // p) ** 2 if n_tokens is None else n_tokens
    patch_proj = (p * p * c) * d  # E in Equation 1 (the paper writes no bias)
    cls = d
    pos = (n + 1) * d  # E_pos in R^{(N+1) x D}
    msa = 4 * d * d + 4 * d  # Wq, Wk, Wv, Wo plus biases
    mlp_p = 2 * d * mlp + mlp + d
    ln = 2 * 2 * d  # two LayerNorms per block, gain and bias each
    block = msa + mlp_p + ln
    final_ln = 2 * d  # LN in Equation 4
    return dict(patch_proj=patch_proj, cls=cls, pos=pos, blocks=layers * block,
                block=block, final_ln=final_ln,
                total=patch_proj + cls + pos + layers * block + final_ln)


# (name, layers L, hidden size D, MLP size, patch size) from Table 1 and Section 4.1
for name, l, d, mlp, p, image in [("ViT-B/16", 12, 768, 3072, 16, 224),
                                  ("ViT-L/16", 24, 1024, 4096, 16, 224),
                                  ("ViT-H/14", 32, 1280, 5120, 14, 224)]:
    r = vit_params(l, d, mlp, p, image=image)
    print(f"{name}: one block {r['block']:,}, all blocks {r['blocks']:,}, "
          f"patch proj {r['patch_proj']:,}, pos {r['pos']:,}, total {r['total']:,} "
          f"({r['total'] / 1e6:.1f}M)")

print("\nViT-B/16 at 384x384 (576 patches): position table grows, nothing else does")
r224 = vit_params(12, 768, 3072, 16, image=224)
r384 = vit_params(12, 768, 3072, 16, image=384)
print(f"  pos params {r224['pos']:,} -> {r384['pos']:,}; "
      f"total {r224['total']:,} -> {r384['total']:,}")

print("\nA head for K=1000 classes adds D*K + K weights; Section 3.2 zero-initializes it")
for name, d in [("ViT-B/16", 768), ("ViT-L/16", 1024), ("ViT-H/14", 1280)]:
    print(f"  {name}: {d * 1000 + 1000:,}")
