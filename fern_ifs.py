"""
IFS code for a Fern (Table III.3)
Two algorithms:
  1. Deterministic algorithm (set iteration)
  2. Random algorithm (chaos game)

Each map is w(x, y) = (a*x + b*y + e, c*x + d*y + f)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# IFS table:  w     a      b      c      d    e     f      p
# ---------------------------------------------------------------
IFS = [
    # a,     b,     c,    d,    e,   f,    p
    (0.00,  0.00,  0.00, 0.16, 0.0, 0.00, 0.01),   # w1
    (0.85,  0.04, -0.04, 0.85, 0.0, 1.60, 0.85),   # w2
    (0.20, -0.26,  0.23, 0.22, 0.0, 1.60, 0.07),   # w3
    (-0.15, 0.28,  0.26, 0.24, 0.0, 0.44, 0.07),   # w4
]


def apply_map(params, pts):
    """Apply affine map w(x,y) = (ax+by+e, cx+dy+f) to an (N,2) array."""
    a, b, c, d, e, f, _ = params
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack((a * x + b * y + e, c * x + d * y + f))


# ---------------------------------------------------------------
# Algorithm 1: Deterministic (set iteration)
# A_{n+1} = w1(A_n) U w2(A_n) U w3(A_n) U w4(A_n)
# ---------------------------------------------------------------
def deterministic_ifs(iterations=9, start=None):
    pts = np.array([[0.0, 0.0]]) if start is None else start
    for _ in range(iterations):
        pts = np.vstack([apply_map(w, pts) for w in IFS])
    return pts


# ---------------------------------------------------------------
# Algorithm 2: Random (chaos game)
# Pick a map w_i with probability p_i and apply it to the current point
# ---------------------------------------------------------------
def random_ifs(n_points=100000, discard=20, seed=None):
    rng = np.random.default_rng(seed)
    probs = np.array([w[6] for w in IFS])
    probs = probs / probs.sum()

    x, y = 0.0, 0.0
    out = np.empty((n_points, 2))
    choices = rng.choice(len(IFS), size=n_points + discard, p=probs)

    for i, k in enumerate(choices):
        a, b, c, d, e, f, _ = IFS[k]
        x, y = a * x + b * y + e, c * x + d * y + f
        if i >= discard:                       # skip transient points
            out[i - discard] = (x, y)
    return out


# ---------------------------------------------------------------
# Main: run both and plot side by side
# ---------------------------------------------------------------
if __name__ == "__main__":
    det_pts = deterministic_ifs(iterations=9)          # 4^9 = 262144 points
    rnd_pts = random_ifs(n_points=100000, seed=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].scatter(det_pts[:, 0], det_pts[:, 1], s=0.2, color="darkgreen")
    axes[0].set_title(f"Deterministic Algorithm ({len(det_pts)} points)")

    axes[1].scatter(rnd_pts[:, 0], rnd_pts[:, 1], s=0.2, color="forestgreen")
    axes[1].set_title(f"Random Algorithm / Chaos Game ({len(rnd_pts)} points)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle("Fern using IFS")
    plt.tight_layout()
    plt.show()
