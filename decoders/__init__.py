"""
Neural decoders subpackage.

Re-exports all public symbols so existing ``from decoders import X`` imports
continue to work unchanged.
"""

from decoders.base import (
    Decoder,
    _compute_poisson_log_likelihoods,
    _validate_decode_inputs,
)
from decoders.direction import (
    PopulationVectorDecoder,
    MaximumLikelihoodDecoder,
    NaiveBayesDecoder,
    get_decoder,
)
from decoders.kalman import KalmanFilterDecoder
from decoders.evaluation import (
    evaluate_decoder,
    compare_decoders,
)

__all__ = [
    'Decoder',
    '_compute_poisson_log_likelihoods',
    '_validate_decode_inputs',
    'PopulationVectorDecoder',
    'MaximumLikelihoodDecoder',
    'NaiveBayesDecoder',
    'get_decoder',
    'KalmanFilterDecoder',
    'evaluate_decoder',
    'compare_decoders',
]
