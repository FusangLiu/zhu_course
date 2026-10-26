"""Editable reaction kinetics. Argument order follows the original CM_Rate.m:602."""


def sbpase(vmax, sbp, pi, s7p, keq, km, ki):
    """SBP ⇌ S7P + Pi, including reverse flux and Pi inhibition. Rate units: mM/s."""
    numerator = sbp - pi * s7p / keq
    denominator = sbp + km * (1 + pi / ki)
    return vmax * numerator / denominator
