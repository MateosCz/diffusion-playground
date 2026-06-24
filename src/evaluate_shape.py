
import itertools
import random
 
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import ks_2samp, wasserstein_distance
 
import lightning as L  # noqa: F401  (kept for parity with training env)
from src.lit.litGNN import LitVanillaGNN
from src.diffusion import TDMDiffusion
from src.nn.vanillaGNN import TDM_VanillaGNN  # noqa: F401
import src.data as data
 
 
# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
 
 
def split_graphs(positions, graph_nodes):
    """Split concatenated node positions (sum_nodes, 2) into a list of per-graph
    (Ni, 2) clouds. Assumes standard PyG batch ordering (graphs concatenated in
    `graph_nodes` order). The assert fails loudly if `ft_list[-1]` ever comes back
    with an unexpected shape (extra trajectory/velocity axis, etc.)."""
    arr = _to_np(positions)
    assert arr.ndim == 2 and arr.shape[1] == 2, \
        f"expected (sum_nodes, 2) positions, got shape {arr.shape}"
    assert arr.shape[0] == sum(graph_nodes), \
        f"position rows {arr.shape[0]} != total nodes {sum(graph_nodes)}"
    idx = np.cumsum([0] + list(graph_nodes))
    return [arr[idx[i]:idx[i + 1]] for i in range(len(graph_nodes))]
 
 
# ---------------------------------------------------------------------------
# canonicalization (period-aware, cross-cell safe)
# ---------------------------------------------------------------------------
def _unwrap_axis(x, L):
    """Un-wrap one periodic coordinate by cutting at the largest empty gap.
    Returns (un-wrapped coords with the arc starting at 0, size of that gap)."""
    x = np.mod(x, L)
    xs = np.sort(x)
    gaps = np.append(np.diff(xs), xs[0] + L - xs[-1])     # last entry = wrap-around gap
    k = int(np.argmax(gaps))
    origin = xs[0] if k == len(xs) - 1 else xs[k + 1]     # arc starts just after the gap
    return np.mod(x - origin, L), gaps[k]
 
 
def detorus_center(p, L, min_gap_frac=0.05):
    """Largest-gap un-wrap per axis (handles cross-cell shapes), then center.
    valid=False when some axis has no clear empty band: the shape genuinely
    spans the cell and cannot be un-wrapped to a single triangle."""
    cols, valid = [], True
    for d in range(p.shape[1]):
        u, gap = _unwrap_axis(p[:, d], L)
        if gap < min_gap_frac * L:
            valid = False
        cols.append(u)
    q = np.stack(cols, axis=1)
    return q - q.mean(0), valid
 
 
def canon(p, L):
    """De-torus + center + RMS-normalize. Returns None for cell-spanning/invalid."""
    q, valid = detorus_center(p, L)
    if not valid:
        return None
    rms = np.sqrt((q ** 2).sum(1).mean())
    return q / (rms + 1e-12)                               # RMS radius = 1
 
 
# ---------------------------------------------------------------------------
# triangle-fit residual
# ---------------------------------------------------------------------------
def _pt_seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(P - proj, axis=1)
 
 
def triangle_residual(p, L):
    """Mean point-to-boundary distance for the best-fit (max-area) triangle.
    Scale/rotation/translation invariant after canon(); ~0 for a clean triangle.
    Returns np.nan for invalid / un-fittable clouds."""
    q = canon(p, L)
    if q is None:
        return np.nan
    try:
        hull = q[ConvexHull(q).vertices]
    except Exception:
        return np.nan
    if len(hull) < 3:
        return np.nan
 
    # the 3 hull vertices enclosing the most area = triangle corners
    best, tri = -1.0, None
    for i, j, k in itertools.combinations(range(len(hull)), 3):
        a, b, c = hull[i], hull[j], hull[k]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area > best:
            best, tri = area, (a, b, c)
 
    a, b, c = tri
    d = np.minimum.reduce([_pt_seg_dist(q, a, b),
                           _pt_seg_dist(q, b, c),
                           _pt_seg_dist(q, c, a)])
    return d.mean()
 
 
def residuals(clouds, L, name=""):
    r = np.array([triangle_residual(c, L) for c in clouds])
    finite = np.isfinite(r)
    print(f"  [{name}] dropped {(~finite).sum()}/{len(r)} as cell-spanning/invalid")
    return r[finite]
 
 
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
 
    # ---- load checkpoints --------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_no_zero_cog = LitVanillaGNN.load_from_checkpoint(
        "checkpoints/20260623_173456/ptiGNN_shapes_triangle_no_zero_cog/last.ckpt",
        weights_only=False,
        map_location=DEVICE,
    )
    gnn_zero_cog_score = LitVanillaGNN.load_from_checkpoint(
        "checkpoints/20260623_173456/ptiGNN_shapes_triangle_zero_cog_score/last.ckpt",
        weights_only=False,
        map_location=DEVICE,
    )
    gnn_no_zero_cog_score = LitVanillaGNN.load_from_checkpoint(
        "checkpoints/20260623_173456/ptiGNN_shapes_triangle_no_zero_cog_score/last.ckpt",
        weights_only=False,
        map_location=DEVICE,
    )
    gnn_no_zero_cog.eval()
    gnn_zero_cog_score.eval()
    gnn_no_zero_cog_score.eval()
 
    # Keep diffusion + score nets on the same device. sample_backward_graph
    # creates graph/time tensors on CUDA when available.
    gnn_no_zero_cog.to(DEVICE)
    gnn_zero_cog_score.to(DEVICE)
    gnn_no_zero_cog_score.to(DEVICE)

    tdm_model_no_zero_cog = TDMDiffusion(
        dim=2, integrator_type="Euler",
        simplified_param=True, zero_cog=False, zero_cog_score=False,
    ).to(DEVICE)
 
    tdm_model = TDMDiffusion(
        dim=2, integrator_type="Euler",
        simplified_param=True, zero_cog=True, zero_cog_score=True,
    ).to(DEVICE)
    tdm_model_no_zero_cog_score = TDMDiffusion(
        dim=2, integrator_type="Euler",
        simplified_param=True, zero_cog=True, zero_cog_score=False,
    ).to(DEVICE)
 
    # ---- sample both groups ------------------------------------------------
    graph_num = 300
    graph_nodes = [random.randint(26, 32) for _ in range(graph_num)]
 
    sample_kwargs = dict(
        fT_prior_kw="uniform",
        vT_prior_kw="stdGauss",
        graph_nodes=graph_nodes,
        data_dim=2,
        sample_trajectory=True,
        n_steps=20,
        exponential_integration=True,
        probability_flow=False,
        predictor_corrector=True,
        predictor_corrector_n_steps=20,
        only_correct_vt=True,
        tau=1e-2,
    )
 
    (ft_list, vt_list, t_list) = tdm_model.sample_backward_graph(
        tdm_score_fn=gnn_zero_cog_score.forward_from_data, **sample_kwargs
    )
    (ft_list_no, vt_list_no, t_list_no) = tdm_model_no_zero_cog_score.sample_backward_graph(
        tdm_score_fn=gnn_no_zero_cog_score.forward_from_data, **sample_kwargs
    )
    (ft_list_no_zero_cog, vt_list_no_zero_cog, t_list_no_zero_cog) = tdm_model_no_zero_cog.sample_backward_graph(
        tdm_score_fn=gnn_no_zero_cog.forward_from_data, **sample_kwargs
    )
 
    f0 = ft_list[-1]                  # generated shapes at t = 0
    f0_no_zero_cog_score = ft_list_no[-1]
    f0_no_zero_cog = ft_list_no_zero_cog[-1]
 
    # ---- ground-truth triangles -------------------------------------------
    # Shapes_Dataset returns (points, corner_mask) with points in [0, 1).
    # Map points -> [-pi, pi) with pos_to_angle so GT matches the angle space
    # the model is sampled in, and every source shares one period L = 2*pi.
    gt_ds = data.Shapes_Dataset(shape_types=["triangle"], num_points=64, seed=0)
    gt_clouds = []
    for i in range(min(graph_num, len(gt_ds))):
        pts, _corner_mask = gt_ds[i]                        # (N, 2) in [0, 1)
        gt_clouds.append(_to_np(data.pos_to_angle(pts)))    # -> [-pi, pi)
 
    # ---- residuals (all sources share one period) -------------------------
    L = 2 * np.pi      # generated and GT both in [-pi, pi)
 
    g_a = split_graphs(f0, graph_nodes)                    # zero_cog_score = True
    g_b = split_graphs(f0_no_zero_cog_score, graph_nodes)  # zero_cog_score = False
    g_c = split_graphs(f0_no_zero_cog, graph_nodes)  # zero_cog = False
    print("=== validity (drop) counts ===")
    r_a = residuals(g_a, L, name="zero_cog_score")
    r_b = residuals(g_b, L, name="no_zero_cog_score")
    r_c = residuals(g_c, L, name="no_zero_cog")
    r_gt = residuals(gt_clouds, L, name="GT")
 
    # ---- report ------------------------------------------------------------
    def report(name, r, ref):
        print(f"{name:14s}  n={len(r):3d}  median={np.median(r):.4f}  "
              f"IQR=[{np.percentile(r, 25):.4f},{np.percentile(r, 75):.4f}]  "
              f"W1(vs GT)={wasserstein_distance(r, ref):.4f}")
 
    print("\n=== triangle-fit residual (lower = more triangle-like) ===")
    report("GT", r_gt, r_gt)
    report("zero_cog_score", r_a, r_gt)
    report("no_zero_cog_score", r_b, r_gt)
    report("no_zero_cog", r_c, r_gt)
 
    print(f"\nKS(zero_cog_score vs no_zero_cog_score): D={ks_2samp(r_a, r_b).statistic:.3f}  "
          f"p={ks_2samp(r_a, r_b).pvalue:.3g}")
    print(f"KS(zero_cog_score vs GT):          D={ks_2samp(r_a, r_gt).statistic:.3f}")
    print(f"KS(no_zero_cog_score vs GT):       D={ks_2samp(r_b, r_gt).statistic:.3f}")
    print(f"KS(no_zero_cog vs GT):       D={ks_2samp(r_c, r_gt).statistic:.3f}")
 
    # ---- one-glance plot ---------------------------------------------------
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, max(r_a.max(), r_b.max(), r_c.max(), r_gt.max()), 40)
    # plt.hist(r_gt, bins, alpha=.5, density=True, label="GT")
    plt.hist(r_a, bins, alpha=.5, density=True, label="zero_cog_score")
    plt.hist(r_b, bins, alpha=.5, density=True, label="no_zero_cog_score")
    plt.hist(r_c, bins, alpha=.5, density=True, label="no_zero_cog")
    plt.xlabel("triangle-fit residual")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("triangle_residual_compare.png", dpi=150)
    print("\nsaved triangle_residual_compare.png")
 