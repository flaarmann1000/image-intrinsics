"""
Laffont & Bazin, ICCV 2015
"Intrinsic Decomposition of Image Sequences from Local Temporal Variations"
https://doi.org/10.1109/ICCV.2015.56

Pipeline summary
----------------
Given T frames I(p,t) = R(p) * S(p,t), solve for shared reflectance R and
per-frame shading S using:
1) local patch constraints with IRLS,
2) long-range pairwise constraints,
3) regularized global sparse solve per channel.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.cluster import KMeans


def decompose(images_np, patch_size=3, n_irls=5,
              n_clusters=50, n_pairs_per_cluster=20,
              gamma_pair=0.1, gamma_reg=1e-4):
    """
    Parameters
    ----------
    images_np           : list of ndarray [H, W, 3] uint8
    patch_size          : spatial patch side-length (paper uses 3)
    n_irls              : IRLS iterations for robust estimation
    n_clusters          : K-means clusters for consistent pair selection
    n_pairs_per_cluster : pixel pairs sampled per cluster
    gamma_pair          : weight for long-range pairwise energy
    gamma_reg           : weight for chromaticity regularization

    Returns
    -------
    reflectance : ndarray [H, W, 3] float in [0, 1]
    shadings    : list of ndarray [H, W] float, length T
    """
    T = len(images_np)
    H, W = images_np[0].shape[:2]
    n_pixels = H * W

    imgs = np.stack([img.astype(np.float64) / 255.0 for img in images_np], axis=0)
    R_flat = np.zeros((n_pixels, 3), dtype=np.float64)

    for c in range(3):
        print(f"  channel {c} ...")
        imgs_c = imgs[:, :, :, c]

        Q_local = _build_local_hessian(imgs_c, patch_size, n_irls)
        pairs = _select_pairs(imgs_c, n_clusters, n_pairs_per_cluster)
        Q_pair = _build_pair_hessian(pairs, imgs_c, n_irls, n_pixels)

        chroma = (3.0 * imgs_c / (imgs.sum(axis=-1) + 1e-8)).mean(axis=0).ravel()

        n_pairs = max(len(pairs), 1)
        Q = (Q_local / n_pixels
             + gamma_pair * Q_pair / n_pairs
             + T * gamma_reg * sparse.eye(n_pixels, format="csr"))
        rhs = T * gamma_reg * chroma

        # Stabilize the linear system for CG in finite precision.
        Q_csr = 0.5 * (Q.tocsr() + Q.tocsr().T)
        mean_abs_diag = float(np.mean(np.abs(Q_csr.diagonal())))
        jitter = max(1e-10, 1e-6 * mean_abs_diag)
        Q_csr = Q_csr + jitter * sparse.eye(n_pixels, format="csr")

        diag = np.maximum(np.abs(Q_csr.diagonal()), jitter)
        M = LinearOperator((n_pixels, n_pixels), matvec=lambda v: v / diag)

        maxiter = max(1000, min(5000, n_pixels // 2))
        x, info = cg(Q_csr, rhs, M=M, x0=chroma,
                     maxiter=maxiter, rtol=1e-5, atol=0.0)

        if info > 0:
            Q_retry = Q_csr + 10.0 * jitter * sparse.eye(n_pixels, format="csr")
            diag_retry = np.maximum(np.abs(Q_retry.diagonal()), 10.0 * jitter)
            M_retry = LinearOperator((n_pixels, n_pixels),
                                     matvec=lambda v: v / diag_retry)
            x, info = cg(Q_retry, rhs, M=M_retry, x0=x,
                         maxiter=2 * maxiter, rtol=3e-5, atol=0.0)

        if info != 0:
            print(f"    CG did not fully converge (info={info})")

        R_flat[:, c] = x

    reflectance = np.clip(R_flat, 0.0, 1.0).reshape(H, W, 3)

    shadings = []
    for t in range(T):
        S = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)
        for c in range(3):
            mask = reflectance[:, :, c] > 1e-4
            S[mask] += imgs[t, :, :, c][mask] / reflectance[:, :, c][mask]
            cnt[mask] += 1.0
        shadings.append(S / np.maximum(cnt, 1.0))

    return reflectance, shadings


def _build_local_hessian(imgs_c, patch_size, n_irls):
    """IRLS for local patch constraints on one channel."""
    T, H, W = imgs_c.shape
    N = patch_size ** 2
    h = patch_size // 2
    n_pixels = H * W
    eps = 1e-8
    max_w = 1e3

    imgs_pad = np.pad(imgs_c, ((0, 0), (h, h), (h, h)), mode="edge")

    patches = np.zeros((T, n_pixels, N), dtype=np.float64)
    pixel_indices = np.zeros((n_pixels, N), dtype=np.int64)

    rr, cc = np.mgrid[0:H, 0:W]
    n = 0
    for dy in range(patch_size):
        for dx in range(patch_size):
            for t in range(T):
                patches[t, :, n] = imgs_pad[t, dy:dy + H, dx:dx + W].ravel()
            rr_cl = np.clip(rr + (dy - h), 0, H - 1)
            cc_cl = np.clip(cc + (dx - h), 0, W - 1)
            pixel_indices[:, n] = (rr_cl * W + cc_cl).ravel()
            n += 1

    weights = np.ones((T, n_pixels), dtype=np.float64)

    for _ in range(n_irls):
        norms = np.maximum(np.linalg.norm(patches, axis=2), eps)
        u = patches / norms[:, :, None]

        W_total = weights.sum(axis=0)
        wu = weights[:, :, None] * u
        wuu = np.einsum("tin,tim->inm", wu, u)

        I_N = np.eye(N)
        H_mats = W_total[:, None, None] * I_N[None] - wuu

        _, eigvecs = np.linalg.eigh(H_mats)
        R_i = eigvecs[:, :, 0]

        u_dot_R = np.einsum("tin,in->ti", u, R_i)
        e = np.maximum(1.0 - u_dot_R ** 2, eps)
        weights = np.clip(1.0 / (2.0 * np.sqrt(e)), 0.0, max_w)

    row_l, col_l, val_l = [], [], []
    for j in range(N):
        for k in range(N):
            row_l.append(pixel_indices[:, j])
            col_l.append(pixel_indices[:, k])
            val_l.append(H_mats[:, j, k])

    return sparse.coo_matrix(
        (np.concatenate(val_l), (np.concatenate(row_l), np.concatenate(col_l))),
        shape=(n_pixels, n_pixels),
    ).tocsr()


def _select_pairs(imgs_c, n_clusters, n_pairs_per_cluster, seed=42):
    """K-means on normalized temporal profiles, then random in-cluster pairs."""
    T, H, W = imgs_c.shape
    n_pixels = H * W

    profiles = imgs_c.reshape(T, n_pixels).T
    norms = np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    normalized = profiles / norms

    k = min(n_clusters, max(1, n_pixels // 4))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(normalized)

    rng = np.random.default_rng(seed)
    pairs = set()
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) < 2:
            continue
        n = min(n_pairs_per_cluster, len(members) * (len(members) - 1) // 2)
        for _ in range(n):
            i, j = rng.choice(len(members), 2, replace=False)
            p, q = int(members[i]), int(members[j])
            pairs.add((min(p, q), max(p, q)))

    return list(pairs)


def _build_pair_hessian(pairs, imgs_c, n_irls, n_pixels):
    """IRLS for long-range pairwise constraints."""
    if not pairs:
        return sparse.csr_matrix((n_pixels, n_pixels))

    T = imgs_c.shape[0]
    profiles = imgs_c.reshape(T, n_pixels)
    eps = 1e-8
    max_w = 1e3

    row_l, col_l, val_l = [], [], []

    for p, q in pairs:
        Ip = profiles[:, p]
        Iq = profiles[:, q]
        wt = np.ones(T, dtype=np.float64)

        for _ in range(n_irls):
            sw = np.sqrt(wt)
            A = np.column_stack([sw * Iq, -sw * Ip])
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            R_pq = Vt[-1]

            res = Iq * R_pq[0] - Ip * R_pq[1]
            wt = np.clip(1.0 / (2.0 * np.maximum(np.abs(res), eps)), 0.0, max_w)

        M = sum(wt[t] * np.outer([Iq[t], -Ip[t]], [Iq[t], -Ip[t]]) for t in range(T))

        for a, pa in enumerate([p, q]):
            for b, pb in enumerate([p, q]):
                row_l.append(pa)
                col_l.append(pb)
                val_l.append(M[a, b])

    return sparse.coo_matrix((val_l, (row_l, col_l)),
                             shape=(n_pixels, n_pixels)).tocsr()
