import numpy as np


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def build_classifier(prompt_embs):
    """Return a (n_classes, d) matrix of unit-length rows.

    prompt_embs: (n_prompts, n_classes, d), unnormalized text embeddings.
    Normalize each embedding, average over the prompts, normalize again.
    """
    return _unit(_unit(prompt_embs).mean(axis=0))


def zero_shot_probs(image_embs, classifier, t):
    """Return (n_images, n_classes) class probabilities.

    image_embs: (n_images, d), unnormalized. classifier: (n_classes, d), unit rows.
    t: log of the logit scale, so the scale is exp(t).
    """
    logits = np.exp(t) * (_unit(image_embs) @ classifier.T)
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)
