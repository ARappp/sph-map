"""Points of a cubic lattice inside a sphere, decided with exact integer arithmetic.
ball(r, step) returns all p = step * (i, j, k), with integers i, j, k and i^2 + j^2 + k^2 <= n^2, n = r/step.
Points on the surface are always included, so the set is symmetric under all sign changes and permutations and the
same on every computer (a floating-point test |p| <= r would include or exclude surface points by rounding)."""
import numpy as np
def ball(r, step=0.1):
    n = int(round(r / step)); assert abs(n * step - r) < 1e-9, "r must be a multiple of step"
    g = np.arange(-n, n + 1); I = np.array([(a, b, c) for a in g for b in g for c in g])
    return step * I[(I ** 2).sum(1) <= n * n]
