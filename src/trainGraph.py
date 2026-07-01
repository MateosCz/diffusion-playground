# src/trainGraph.py
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.data import Checkerboard_Dataset, TorusLieWrapper, AngleTorusWrapper, Pacman_Dataset, Shapes_Dataset, PyGGraphWrapper
from src.nn.vanillaGNN import TDM_VanillaGNN
from src.diffusion.tdm import TDMDiffusion
from src.device import get_default_device
from torch_geometric.loader import DataLoader as PyGDataLoader

def time_loss_weight(t, total_time):
    t_norm = (t / total_time).clamp(0.0, 1.0)
    return (1.0 - t_norm).clamp_min(0.05)


def weighted_score_loss(pred_score, target_score, t, total_time):
    '''
    Parameters:
        pred_score: (N_total, 2)
        target_score: (N_total, 2)
        t: (N_total, 1)
        total_time: float
    Returns:
        loss: scalar
    '''
    weight = time_loss_weight(t, total_time)
    while weight.ndim < pred_score.ndim:
        weight = weight.unsqueeze(-1)
    return torch.mean(weight * (pred_score - target_score) ** 2)


def score_diagnostics(pred_score, target_score, t=None, total_time=None):
    pred_flat = pred_score.reshape(pred_score.shape[0], -1)
    target_flat = target_score.reshape(target_score.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    mse = torch.nn.functional.mse_loss(pred_score, target_score)
    zero_mse = torch.mean(target_score**2)
    diagnostics = {
        "mse": mse.item(),
        "zero_mse": zero_mse.item(),
        "improvement": (1.0 - mse / zero_mse.clamp_min(1e-12)).item(),
        "pred_std": pred_score.std().item(),
        "target_std": target_score.std().item(),
        "cosine": cosine.item(),
    }
    if t is not None and total_time is not None:
        weighted_mse = weighted_score_loss(pred_score, target_score, t, total_time)
        weighted_zero_mse = weighted_score_loss(torch.zeros_like(target_score), target_score, t, total_time)
        diagnostics["weighted_mse"] = weighted_mse.item()
        diagnostics["weighted_improvement"] = (
            1.0 - weighted_mse / weighted_zero_mse.clamp_min(1e-12)
        ).item()
        diagnostics["mean_t"] = t.mean().item()
    else:
        diagnostics["weighted_mse"] = mse.item()
        diagnostics["weighted_improvement"] = diagnostics["improvement"]
        diagnostics["mean_t"] = float("nan")
    return {
        **diagnostics,
    }


@torch.no_grad()
def evaluate_score_model(model, diffusion, loader, device, total_time, max_batches=4):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0
    total_pred_std = 0.0
    total_target_std = 0.0
    total_zero_mse = 0.0
    total_improvement = 0.0
    total_weighted_mse = 0.0
    total_weighted_improvement = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        (v_t, batch_graph_noised), target_score, t_scalar = diffusion.sample_forward_graph(
            graph=batch,
            total_time=total_time,
            t_dist_kw="uniform",
            v0_dist_kw="zero",
            return_time=True,
        )
        pred_score = model.forward_from_data(batch_graph_noised, v_t, t_scalar) # (N_total, 2)
        t_all = t_scalar[batch.batch] # (N_total,)
        loss = weighted_score_loss(pred_score, target_score, t_all, total_time)
        diagnostics = score_diagnostics(pred_score, target_score, t_all, total_time)

        total_loss += loss.item()
        total_cosine += diagnostics["cosine"]
        total_pred_std += diagnostics["pred_std"]
        total_target_std += diagnostics["target_std"]
        total_zero_mse += diagnostics["zero_mse"]
        total_improvement += diagnostics["improvement"]
        total_weighted_mse += diagnostics["weighted_mse"]
        total_weighted_improvement += diagnostics["weighted_improvement"]
        n_batches += 1

        if n_batches >= max_batches:
            break

    if was_training:
        model.train()

    return {
        "loss": total_loss / n_batches,
        "cosine": total_cosine / n_batches,
        "pred_std": total_pred_std / n_batches,
        "target_std": total_target_std / n_batches,
        "zero_mse": total_zero_mse / n_batches,
        "improvement": total_improvement / n_batches,
        "weighted_mse": total_weighted_mse / n_batches,
        "weighted_improvement": total_weighted_improvement / n_batches,
    }


@torch.no_grad()
def evaluate_score_model_fixed_times(
    model,
    diffusion,
    loader,
    device,
    total_time,
    fixed_times=(0.1, 0.2, 0.5, 1.0),
):
    was_training = model.training
    model.eval()
    batch = next(iter(loader)).to(device)
    summaries = []

    for fixed_t in fixed_times:
        (v_t, batch_graph_noised), target_score, t_scalar = diffusion.sample_forward_graph(
            graph=batch,
            total_time=total_time,
            t_dist_kw="constant",
            constant_t=fixed_t,
            v0_dist_kw="zero",
            return_time=True,
        )
        pred_score = model.forward_from_data(batch_graph_noised, v_t, t_scalar) # (N_total, 2)
        t_all = t_scalar[batch.batch] # (N_total,)
        diagnostics = score_diagnostics(pred_score, target_score, t_all, total_time)
        summaries.append(
            f"t={fixed_t:.1f}:imp={diagnostics['improvement']:.3f},"
            f"cos={diagnostics['cosine']:.3f},std={diagnostics['pred_std']:.3f}"
        )

    if was_training:
        model.train()

    return " | ".join(summaries)



def main():
    # -----------------------
    # Config (simple defaults)
    # -----------------------
    device = get_default_device()
    batch_size = 32
    n_epoch = 200
    lr = 1e-3
    total_time = 2.0
    pt_invariant = True
    # base_ds_kw = "triangle"
    base_ds_kw = "mix"

    num_mp_layers = 6
    node_feat_dim = 2
    edge_fourier_bands = 8
    v_dim = 2

    # data shape: each sample -> (dim,)
    dim = 2
    # model
    time_embedding_half_dim = 32  # must be even
    time_embedding_scale = 1.0
    position_fourier_bands = 8
    with_sincos_position = True
    only_sincos_position = True
    t_dist_kw = "uniform"
    hidden_dim = [512,512]
    output_dim = dim
    # dataset
    if base_ds_kw == "triangle":
        base_ds = Shapes_Dataset(
            num_points=32,
            dataset_size=10000,
            shape_types=["triangle"],
            centered=pt_invariant
        )
    elif base_ds_kw == "mix":
        base_ds = Shapes_Dataset(
            num_points=32,
            dataset_size=20000,
            shape_types=["triangle", "rectangle", "star"],
            centered=pt_invariant
        )
    graph_ds = PyGGraphWrapper(base_ds, num_points_range=(26,32))
    loader = PyGDataLoader(graph_ds, batch_size=batch_size, shuffle=True)
    val_graph_ds = PyGGraphWrapper(base_ds, num_points_range=(26,32))
    val_loader = PyGDataLoader(val_graph_ds, batch_size=batch_size, shuffle=False)
    # diffusion + score model
    diffusion = TDMDiffusion(dim=dim, integrator_type="Euler", simplified_param=True).to(device)
    model = TDM_VanillaGNN(
        node_feat_dim=node_feat_dim,
        edge_fourier_bands=edge_fourier_bands,
        v_dim=v_dim,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        time_embedding_half_dim=time_embedding_half_dim,
        position_fourier_bands=position_fourier_bands,
        with_sincos_position=with_sincos_position,
        only_sincos_position=only_sincos_position,
        pt_invariant=pt_invariant,
        zero_cog=pt_invariant,
        output_dim=output_dim,
        total_time=total_time,
        time_embedding_scale=time_embedding_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
    # -----------------------
    # Training loop
    # -----------------------
    model.train()
    epoch_losses = []
    for epoch in range(1, n_epoch + 1):
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epoch}", leave=False)
        for batch_idx, batch_graph in enumerate(pbar, start=1):
            # batch: (N_total)
            batch_graph = batch_graph.to(device)
            # sample noised state + target score
            # latents = (v_t, f_t), each (N_total, 2)
            # t_scalar: (num_graphs, 1)
            (v_t, batch_graph_noised), target_score, t_scalar = diffusion.sample_forward_graph(
                graph=batch_graph,
                total_time=total_time,
                t_dist_kw=t_dist_kw,
                v0_dist_kw="zero",
                return_time=True,
                zero_cog=pt_invariant
            )
            pred_score = model.forward_from_data(batch_graph_noised, v_t, t_scalar) # (N_total, 2)

            t_all = t_scalar[batch_graph.batch] # (N_total,)

            loss = weighted_score_loss(pred_score, target_score, t_all, total_time)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            diagnostics = score_diagnostics(pred_score.detach(), target_score.detach(), t_all, total_time)
            pbar.set_postfix(
                batch=f"{batch_idx}/{len(loader)}",
                loss=f"{loss.item():.4f}",
                imp=f"{diagnostics['improvement']:.3f}",
                wimp=f"{diagnostics['weighted_improvement']:.3f}",
                cos=f"{diagnostics['cosine']:.3f}",
                t=f"{diagnostics['mean_t']:.2f}",
                pred_std=f"{diagnostics['pred_std']:.2f}",
                target_std=f"{diagnostics['target_std']:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)
        val_metrics = evaluate_score_model(model, diffusion, val_loader, device, total_time)
        scheduler.step()
        print(
            f"Epoch [{epoch:03d}/{n_epoch}]  "
            f"train_loss={avg_loss:.6f}  "
            f"val_loss={val_metrics['loss']:.6f}  "
            f"val_zero={val_metrics['zero_mse']:.6f}  "
            f"val_imp={val_metrics['improvement']:.3f}  "
            f"val_wimp={val_metrics['weighted_improvement']:.3f}  "
            f"val_cos={val_metrics['cosine']:.3f}  "
            f"pred_std={val_metrics['pred_std']:.3f}  "
            f"target_std={val_metrics['target_std']:.3f}"
        )
        fixed_time_summary = evaluate_score_model_fixed_times(
            model,
            diffusion,
            val_loader,
            device,
            total_time,
        )
        print(f"  fixed_time_val: {fixed_time_summary}")
    
    # plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker="x", linewidth=1, markersize=5, linestyle="--", color="gold", markeredgecolor="green")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"training_loss_{base_ds_kw}_shapes.png", dpi=150)
    plt.show()
    # save model
    if pt_invariant:
        torch.save(model.state_dict(), f"vanilla_gnn_{base_ds_kw}_shapes_pti.pt")
        print(f"Training done. Saved model to vanilla_gnn_{base_ds_kw}_shapes_pti.pt")
    else:
        torch.save(model.state_dict(), f"vanilla_gnn_{base_ds_kw}.pt")
        print(f"Training done. Saved model to vanilla_gnn_{base_ds_kw}.pt")
if __name__ == "__main__":
    main()