"""
Sierpinski Triangle via Iterated Function System (IFS)
Two algorithms:
  1. Deterministic algorithm (set iteration)
  2. Random algorithm (chaos game)

Each map is w(x, y) = (a*x + b*y + e, c*x + d*y + f)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# IFS table:  w    a    b    c    d    e    f    p
# ---------------------------------------------------------------
IFS = [
    # a,   b,   c,   d,   e,    f,    p
    (0.5, 0.0, 0.0, 0.5,  0.0,  0.0, 0.33),   # w1
    (0.5, 0.0, 0.0, 0.5, 50.0,  0.0, 0.33),   # w2
    (0.5, 0.0, 0.0, 0.5, 50.0, 50.0, 0.34),   # w3
]


def apply_map(params, pts):
    """Apply affine map w(x,y) = (ax+by+e, cx+dy+f) to an (N,2) array."""
    a, b, c, d, e, f, _ = params
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack((a * x + b * y + e, c * x + d * y + f))


# ---------------------------------------------------------------
# Algorithm 1: Deterministic (set iteration)
# A_{n+1} = w1(A_n) U w2(A_n) U w3(A_n)
# ---------------------------------------------------------------
def deterministic_ifs(iterations=9, start=None):
    if start is None:
        start = np.array([[0.0, 0.0]])          # any starting set works
    pts = start
    for _ in range(iterations):
        pts = np.vstack([apply_map(w, pts) for w in IFS])
    return pts


# ---------------------------------------------------------------
# Algorithm 2: Random (chaos game)
# Pick a map w_i with probability p_i and apply it to the current point
# ---------------------------------------------------------------
def random_ifs(n_points=50000, discard=20, seed=None):
    rng = np.random.default_rng(seed)
    probs = np.array([w[6] for w in IFS])
    probs = probs / probs.sum()                 # normalise (0.33+0.33+0.34 = 1)

    x, y = rng.random(), rng.random()
    out = np.empty((n_points, 2))
    choices = rng.choice(len(IFS), size=n_points + discard, p=probs)

    for i, k in enumerate(choices):
        a, b, c, d, e, f, _ = IFS[k]
        x, y = a * x + b * y + e, c * x + d * y + f
        if i >= discard:                        # skip transient points
            out[i - discard] = (x, y)
    return out


# ---------------------------------------------------------------
# Main: run both and plot side by side
# ---------------------------------------------------------------
if __name__ == "__main__":
    det_pts = deterministic_ifs(iterations=9)
    rnd_pts = random_ifs(n_points=50000, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(det_pts[:, 0], det_pts[:, 1], s=0.3, color="darkblue")
    axes[0].set_title(f"Deterministic Algorithm ({len(det_pts)} points)")

    axes[1].scatter(rnd_pts[:, 0], rnd_pts[:, 1], s=0.3, color="darkred")
    axes[1].set_title(f"Random Algorithm / Chaos Game ({len(rnd_pts)} points)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle("Sierpinski Triangle using IFS")
    plt.tight_layout()
    plt.show()
