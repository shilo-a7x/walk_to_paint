"""
Shared MI-via-PCA+binning utility for Lead 2 (GNN bottleneck) diagnostics.

node_mi_structural_embedding.py already implements PCA -> percentile-bin ->
joint-histogram MI, but only for the *symmetric* case where both sides of the
pair are the same scalar node feature (feature(A) vs feature(B)). Lead 2 needs
the *asymmetric* case: a continuous embedding (a GNN's h_v^(1), or a
transformer's hidden state) on one side, and a scalar signal (a +-1 edge sign)
on the other. mi_pca_bins() below generalizes that pattern to this case.

Reuses mi_from_joint / entropy_from_marginal / bfs_frontiers unmodified from
node_mi_structural_embedding.py rather than re-deriving them.
"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import (
    mi_from_joint, entropy_from_marginal, bfs_frontiers,
)

N_BINS_DEFAULT = 5
N_PCA_DEFAULT = 5


def _global_bins(values: np.ndarray, n_bins: int):
    """Global binning used for both the anchor and context side. A variable
    with <= n_bins distinct values (e.g. a +-1 sign, or a degenerate/discrete
    embedding dimension) is kept as its own exact bins (lossless); anything
    with more distinct values is percentile-binned, identical scheme to
    node_mi_structural_embedding.py. Returns (bin_idx int16 array, n_bins_actual)."""
    values = np.asarray(values, dtype=np.float64)
    uniq = np.unique(values)
    if len(uniq) <= n_bins:
        lut = {v: i for i, v in enumerate(uniq)}
        bidx = np.array([lut[v] for v in values], dtype=np.int16)
        return bidx, max(len(uniq), 1)
    edges = np.unique(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 2:
        return np.zeros(len(values), dtype=np.int16), 1
    bidx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return bidx.astype(np.int16), len(edges) - 1


def mi_pca_bins(anchor_embeddings: np.ndarray, context_values: np.ndarray,
                 n_pca: int = N_PCA_DEFAULT, n_bins: int = N_BINS_DEFAULT,
                 pca_random_state: int = 42) -> dict:
    """
    MI(anchor_embeddings, context_values) via the same PCA -> percentile-bin
    -> joint-histogram method as node_mi_structural_embedding.py, generalized
    to an asymmetric anchor (possibly multi-dim continuous embedding) vs.
    context (scalar, e.g. a +-1 sign) pairing.

    anchor_embeddings: (N, D) float array, row-aligned with context_values.
        D==1 is treated as already-scalar (no PCA needed). Non-finite rows
        are dropped.
    context_values: (N,) array.

    Returns a dict:
      "mi_per_component": list of MI(bits), one per PCA component (or a
          single-element list if D==1)
      "mi": max over components (the headline number -- a LOWER BOUND on the
          true joint MI(embedding, context), since each component's MI is
          computed univariately, not jointly across all components at once)
      "best_component": index of the component achieving "mi" (-1 if all nan)
      "nmi": "mi" / H(context) (entropy of context's own marginal) -- nan if
          H(context) ~ 0
      "n_pairs": number of (finite) rows used
      "n_components": number of PCA components actually used
    """
    X = np.asarray(anchor_embeddings, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(context_values, dtype=np.float64)

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    n_pairs = len(y)
    if n_pairs < 2:
        return {"mi_per_component": [], "mi": float("nan"), "best_component": -1,
                "nmi": float("nan"), "n_pairs": n_pairs, "n_components": 0}

    if X.shape[1] == 1:
        comps = X
        n_comp = 1
    else:
        n_comp = min(n_pca, X.shape[1], n_pairs - 1)
        pca = PCA(n_components=n_comp, random_state=pca_random_state)
        comps = pca.fit_transform(X)

    y_bins, n_bins_y = _global_bins(y, n_bins)
    h_context = entropy_from_marginal(np.bincount(y_bins, minlength=n_bins_y))

    mi_per_component = []
    for k in range(n_comp):
        x_bins, n_bins_x = _global_bins(comps[:, k], n_bins)
        combined = x_bins.astype(np.int64) * n_bins_y + y_bins.astype(np.int64)
        joint = np.bincount(combined, minlength=n_bins_x * n_bins_y).reshape(n_bins_x, n_bins_y)
        mi_per_component.append(mi_from_joint(joint))

    finite_mask = [not np.isnan(m) for m in mi_per_component]
    if any(finite_mask):
        best_idx = int(np.nanargmax(mi_per_component))
        mi_best = mi_per_component[best_idx]
    else:
        best_idx = -1
        mi_best = float("nan")
    nmi_best = mi_best / h_context if (h_context > 1e-12 and not np.isnan(mi_best)) else float("nan")

    return {
        "mi_per_component": mi_per_component,
        "mi": mi_best,
        "best_component": best_idx,
        "nmi": nmi_best,
        "n_pairs": n_pairs,
        "n_components": n_comp,
    }
