import numpy as np


def build_classifier(prompt_embs):
    """Return a (n_classes, d) matrix of unit-length rows.

    prompt_embs: (n_prompts, n_classes, d), unnormalized text embeddings.
    Normalize each embedding, average over the prompts, normalize again.
    """
    raise NotImplementedError


def zero_shot_probs(image_embs, classifier, t):
    """Return (n_images, n_classes) class probabilities.

    image_embs: (n_images, d), unnormalized. classifier: (n_classes, d), unit rows.
    t: log of the logit scale, so the scale is exp(t).
    """
    raise NotImplementedError
