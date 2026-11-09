def model_state_bytes(psi, nd, stage, K=12):
    """Bytes of model states (fp16 params, fp16 grads, optimizer state) on one GPU.

    stage 0 = data parallelism, 1 = P_os, 2 = P_os+g, 3 = P_os+g+p.
    """
    raise NotImplementedError


def max_params(mem_bytes, nd, stage, K=12):
    """Largest parameter count whose model states fit in mem_bytes."""
    raise NotImplementedError
