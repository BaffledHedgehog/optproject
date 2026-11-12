"""Certified diameter envelopes; see README_RU.txt for the guarantees."""
from .api import (
    DiameterCertificate, Model, Precision, ResourceLimits,
    certify_joint_diameter, certify_norm_diameter,
)

__all__ = [
    "DiameterCertificate", "Model", "Precision", "ResourceLimits",
    "certify_joint_diameter", "certify_norm_diameter",
]
