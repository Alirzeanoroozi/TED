"""Domain queries (DETR / Mask2Former style): each query predicts one domain mask, CATH label and confidence."""

from .model import DomainQueryModel
from .decode import decode_domains, decode_batch

__all__ = [
    "DomainQueryModel",
    "decode_domains",
    "decode_batch",
]
