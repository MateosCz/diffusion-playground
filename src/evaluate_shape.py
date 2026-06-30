
import itertools
import random
 
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm.auto import tqdm
 
import lightning as L  # noqa: F401  (kept for parity with training env)
from src.lit.litGNN import LitVanillaGNN
from src.diffusion import TDMDiffusion
from src.nn.vanillaGNN import TDM_VanillaGNN  # noqa: F401
import src.dataLib.synthetic as data
from src.device import get_default_device, module_device
 
 
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
 
 
def _max_area_triangle(points):
    """Return (a, b, c) vertices of max-area triangle on the hull points."""
    best, tri = -1.0, None
    for i, j, k in itertools.combinations(range(len(points)), 3):
        a, b, c = points[i], points[j], points[k]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area > best:
            best, tri = area, (a, b, c)
    return tri


def _triangle_from_cloud(p, L):
    """Get canonical cloud + max-area triangle corners. Returns (q, tri) or (None, None)."""
    q = canon(p, L)
    if q is None:
        return None, None
    try:
        hull = q[ConvexHull(q).vertices]
    except Exception:
        return None, None
    if len(hull) < 3:
        return None, None
    tri = _max_area_triangle(hull)
    if tri is None:
        return None, None
    return q, tri


def _pt_triangle_dist(P, tri):
    a, b, c = tri
    return np.minimum.reduce([_pt_seg_dist(P, a, b),
                              _pt_seg_dist(P, b, c),
                              _pt_seg_dist(P, c, a)])


def _rotate(P, theta):
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    return P @ R.T


def superimposed_rms_point_to_edge(p_gen, p_gt, L, n_angles=180):
    """Align generated cloud to GT (rotation + optional mirror), then compute
    RMS distance from generated points to GT triangle edges."""
    q_gen = canon(p_gen, L)
    q_gt, tri_gt = _triangle_from_cloud(p_gt, L)
    if q_gen is None or q_gt is None or tri_gt is None:
        return np.nan

    best = np.inf
    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    # Try no reflection and x-axis reflection in canonical coordinates.
    for reflect in (1.0, -1.0):
        q = q_gen.copy()
        q[:, 0] *= reflect
        for theta in thetas:
            q_aligned = _rotate(q, theta)
            d = _pt_triangle_dist(q_aligned, tri_gt)
            rms = np.sqrt(np.mean(d ** 2))
            if rms < best:
                best = rms
    return best


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
    tri = _max_area_triangle(hull)
    if tri is None:
        return np.nan
 
    a, b, c = tri
    d = np.minimum.reduce([_pt_seg_dist(q, a, b),
                           _pt_seg_dist(q, b, c),
                           _pt_seg_dist(q, c, a)])
    return d.mean()
 
 
def residuals(clouds, L, name="", show_progress=False):
    cloud_iter = tqdm(clouds, desc=f"{name} residuals", leave=False) if show_progress else clouds
    r = np.array([triangle_residual(c, L) for c in cloud_iter])
    finite = np.isfinite(r)
    print(f"  [{name}] dropped {(~finite).sum()}/{len(r)} as cell-spanning/invalid")
    return r[finite]


def paired_superimposed_rms(gen_clouds, gt_clouds, L, name="", n_angles=180, show_progress=False):
    n = min(len(gen_clouds), len(gt_clouds))
    idx_iter = tqdm(range(n), desc=f"{name} superimposed RMS", leave=False) if show_progress else range(n)
    r = np.array([superimposed_rms_point_to_edge(gen_clouds[i], gt_clouds[i], L, n_angles=n_angles)
                  for i in idx_iter])
    finite = np.isfinite(r)
    print(f"  [{name}] superimposed dropped {(~finite).sum()}/{len(r)} invalid pairs")
    return r[finite]
 
 
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
 
    # ---- load checkpoints --------------------------------------------------
    DEVICE = get_default_device()
    print(f"[device] selected default device: {DEVICE}")

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

    print(f"[device] gnn_no_zero_cog: {module_device(gnn_no_zero_cog)}")
    print(f"[device] gnn_zero_cog_score: {module_device(gnn_zero_cog_score)}")
    print(f"[device] gnn_no_zero_cog_score: {module_device(gnn_no_zero_cog_score)}")
    print(f"[device] tdm_model: {module_device(tdm_model)}")
    print(f"[device] tdm_model_no_zero_cog_score: {module_device(tdm_model_no_zero_cog_score)}")
    print(f"[device] tdm_model_no_zero_cog: {module_device(tdm_model_no_zero_cog)}")
 
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
        verbose_device=True,
    )
 
    (ft_list, vt_list, t_list) = tdm_model.sample_backward_graph(
        tdm_score_fn=gnn_zero_cog_score.forward_from_data,
        progress=True,
        progress_desc="sample zero_cog_score",
        **sample_kwargs
    )
    (ft_list_no, vt_list_no, t_list_no) = tdm_model_no_zero_cog_score.sample_backward_graph(
        tdm_score_fn=gnn_no_zero_cog_score.forward_from_data,
        progress=True,
        progress_desc="sample no_zero_cog_score",
        **sample_kwargs
    )
    (ft_list_no_zero_cog, vt_list_no_zero_cog, t_list_no_zero_cog) = tdm_model_no_zero_cog.sample_backward_graph(
        tdm_score_fn=gnn_no_zero_cog.forward_from_data,
        progress=True,
        progress_desc="sample no_zero_cog",
        **sample_kwargs
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
    for i in tqdm(range(min(graph_num, len(gt_ds))), desc="prepare GT clouds", leave=False):
        pts, _corner_mask = gt_ds[i]                        # (N, 2) in [0, 1)
        gt_clouds.append(_to_np(data.pos_to_angle(pts)))    # -> [-pi, pi)
 
    # ---- residuals (all sources share one period) -------------------------
    L = 2 * np.pi      # generated and GT both in [-pi, pi)
 
    g_a = split_graphs(f0, graph_nodes)                    # zero_cog_score = True
    g_b = split_graphs(f0_no_zero_cog_score, graph_nodes)  # zero_cog_score = False
    g_c = split_graphs(f0_no_zero_cog, graph_nodes)  # zero_cog = False
    print("=== validity (drop) counts ===")
    r_a = residuals(g_a, L, name="zero_cog_score", show_progress=True)
    r_b = residuals(g_b, L, name="no_zero_cog_score", show_progress=True)
    r_c = residuals(g_c, L, name="no_zero_cog", show_progress=True)
    r_gt = residuals(gt_clouds, L, name="GT", show_progress=True)
    s_a = paired_superimposed_rms(g_a, gt_clouds, L, name="zero_cog_score", show_progress=True)
    s_b = paired_superimposed_rms(g_b, gt_clouds, L, name="no_zero_cog_score", show_progress=True)
    s_c = paired_superimposed_rms(g_c, gt_clouds, L, name="no_zero_cog", show_progress=True)
 
    # ---- report ------------------------------------------------------------
    report_lines = []

    def log(line):
        print(line)
        report_lines.append(line)

    def report(name, r, ref):
        log(f"{name:14s}  n={len(r):3d}  median={np.median(r):.4f}  "
            f"IQR=[{np.percentile(r, 25):.4f},{np.percentile(r, 75):.4f}]  "
            f"W1(vs GT)={wasserstein_distance(r, ref):.4f}")

    log("\n=== triangle-fit residual (lower = more triangle-like) ===")
    report("GT", r_gt, r_gt)
    report("zero_cog_score", r_a, r_gt)
    report("no_zero_cog_score", r_b, r_gt)
    report("no_zero_cog", r_c, r_gt)
 
    log(f"\nKS(zero_cog_score vs no_zero_cog_score): D={ks_2samp(r_a, r_b).statistic:.3f}  "
        f"p={ks_2samp(r_a, r_b).pvalue:.3g}")
    log(f"KS(zero_cog_score vs GT):          D={ks_2samp(r_a, r_gt).statistic:.3f}")
    log(f"KS(no_zero_cog_score vs GT):       D={ks_2samp(r_b, r_gt).statistic:.3f}")
    log(f"KS(no_zero_cog vs GT):       D={ks_2samp(r_c, r_gt).statistic:.3f}")
    log("\n=== superimposed RMS point-to-GT-edge (lower = better) ===")
    log(f"zero_cog_score     n={len(s_a):3d}  median={np.median(s_a):.4f}  "
        f"IQR=[{np.percentile(s_a, 25):.4f},{np.percentile(s_a, 75):.4f}]")
    log(f"no_zero_cog_score  n={len(s_b):3d}  median={np.median(s_b):.4f}  "
        f"IQR=[{np.percentile(s_b, 25):.4f},{np.percentile(s_b, 75):.4f}]")
    log(f"no_zero_cog        n={len(s_c):3d}  median={np.median(s_c):.4f}  "
        f"IQR=[{np.percentile(s_c, 25):.4f},{np.percentile(s_c, 75):.4f}]")

    report_path = "triangle_residual_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nsaved {report_path}")
 
    # ---- one-glance plot ---------------------------------------------------
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, max(r_a.max(), r_b.max(), r_c.max(), r_gt.max()), 40)
    # Outline-style histograms are easier to compare when distributions overlap.
    plt.hist(r_a, bins, density=True, histtype="step", linewidth=2, label="zero_cog_score")
    plt.hist(r_b, bins, density=True, histtype="step", linewidth=2, label="no_zero_cog_score")
    plt.hist(r_c, bins, density=True, histtype="step", linewidth=2, label="no_zero_cog")
    plt.xlabel("triangle-fit residual")
    plt.ylabel("density")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("triangle_residual_compare.png", dpi=150)
    print("\nsaved triangle_residual_compare.png")

    plt.figure(figsize=(6, 4))
    bins_s = np.linspace(0, max(s_a.max(), s_b.max(), s_c.max()), 40)
    plt.hist(s_a, bins_s, density=True, histtype="step", linewidth=2, label="zero_cog_score")
    plt.hist(s_b, bins_s, density=True, histtype="step", linewidth=2, label="no_zero_cog_score")
    plt.hist(s_c, bins_s, density=True, histtype="step", linewidth=2, label="no_zero_cog")
    plt.xlabel("superimposed RMS point-to-GT-edge")
    plt.ylabel("density")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("triangle_superimposed_rms_compare.png", dpi=150)
    print("saved triangle_superimposed_rms_compare.png")
 