"""Causal stopping certificates from paid first-order source observations.

Every completed block makes one actual extra full LMO call. A separate
known-mu control uses strong-convexity lower bounds without extra LMOs.
No problem/reference objects are accepted by this module.
"""

import math

import numpy as np

from problems import lmo


PROFILES = ("scalar", "block_8", "block_32", "block_128", "known_mu")


class TangentMonitor:
    def __init__(self, radius, profile, mu=None):
        if profile not in PROFILES or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Need a valid radius and certificate profile")
        if profile == "known_mu" and (mu is None or not math.isfinite(mu) or mu <= 0):
            raise ValueError("The known-mu control needs a valid positive strong-convexity input")
        self.radius, self.profile, self.mu = radius, profile, mu
        self.block_size = int(profile.split("_")[1]) if profile.startswith("block") else None
        self.best = -math.inf
        self.block_intercepts = []
        self.block_gradients = []
        self.sources = self.extra_lmo = 0

    def observe(self, x, fx, gradient, scalar_lower):
        """Each call consumes one already-paid source, in chronological order."""
        self.sources += 1
        evaluated = False
        candidate = None
        if self.block_size:
            self.block_intercepts.append(float(fx-gradient@x))
            self.block_gradients.append(gradient.copy())
            if len(self.block_intercepts) == self.block_size:
                mean_gradient = np.mean(self.block_gradients, axis=0)
                atom = lmo(mean_gradient, self.radius)
                self.extra_lmo += 1
                candidate = float(math.fsum(self.block_intercepts)/self.block_size+mean_gradient@atom)
                self.best = max(self.best, candidate)
                self.block_intercepts.clear()
                self.block_gradients.clear()
                evaluated = True
        elif self.profile == "known_mu":
            candidate = float(fx-float(gradient@gradient)/(2*self.mu))
            self.best = max(self.best, candidate)
            evaluated = True
        lower = max(scalar_lower, self.best)
        return {"profile": self.profile, "certificate": max(0.0, float(fx-lower)),
                "lower_bound": lower, "scalar_certificate": max(0.0, float(fx-scalar_lower)),
                "certificate_lmo_evals": self.extra_lmo,
                "candidate_evaluated": evaluated, "candidate_lower_bound": candidate}
