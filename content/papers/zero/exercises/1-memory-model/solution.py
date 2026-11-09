def model_state_bytes(psi, nd, stage, K=12):
    """Bytes of model states (fp16 params, fp16 grads, optimizer state) on one GPU.

    stage 0 = data parallelism, 1 = P_os, 2 = P_os+g, 3 = P_os+g+p.
    """
    params = 2 * psi
    grads = 2 * psi
    opt = K * psi
    if stage >= 1:
        opt = opt / nd
    if stage >= 2:
        grads = grads / nd
    if stage >= 3:
        params = params / nd
    return params + grads + opt


def max_params(mem_bytes, nd, stage, K=12):
    """Largest parameter count whose model states fit in mem_bytes."""
    return mem_bytes / model_state_bytes(1.0, nd, stage, K)
