"Helper."""

import numpy as np


def zero_mean_unit_var_norm(input_values: list[np.ndarray]) -> list[np.ndarray]:
    """
    Every array in the list is normalized to have zero mean and unit variance
    Taken from huggingface to ensure the same behavior across s3prl and huggingface
    Reference: https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L81-L86
    """
    return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]
