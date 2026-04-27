# src/train.py
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.data import Checkerboard_Dataset, TorusLieWrapper, AngleTorusWrapper
from src.scoreNN import TDM_SimpleScoreMLP
from src.diffusion import TDMDiffusion


def time_loss_weight(t, total_time):
    t_norm = (t / total_time).clamp(0.0, 1.0)
    return (1.0 - t_norm).clamp_min(0.05)


def weighted_score_loss(pred_score, target_score, t, total_time):
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

    for f0 in loader:
        f0 = f0.to(device, dtype=torch.float32)
        (v_t, f_t), target_score, t_scalar = diffusion.sample_forward(
            f0=f0,
            total_time=total_time,
            t_dist_kw="uniform",
            v0_dist_kw="zero",
            return_time=True,
        )
        pred_score = model(f_t, v_t, t_scalar)
        loss = diffusion.loss_diffusion(pred_score, target_score, t_scalar)
        diagnostics = score_diagnostics(pred_score, target_score, t_scalar, total_time)

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
    f0 = next(iter(loader)).to(device, dtype=torch.float32)
    summaries = []

    for fixed_t in fixed_times:
        (v_t, f_t), target_score, t_scalar = diffusion.sample_forward(
            f0=f0,
            total_time=total_time,
            t_dist_kw="constant",
            constant_t=fixed_t,
            v0_dist_kw="zero",
            return_time=True,
        )
        pred_score = model(f_t, v_t, t_scalar)
        diagnostics = score_diagnostics(pred_score, target_score, t_scalar, total_time)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 512
    n_epoch = 200
    lr = 1e-3
    total_time = 2.0
    # data shape: each sample -> (dim,)
    dim = 2
    # model
    x_lifting_dim = 64
    time_embedding_half_dim = 32  # must be even
    time_embedding_scale = 1.0
    position_fourier_bands = 8
    t_dist_kw = "quadratic"
    use_weighted_loss = True
    hidden_dim = [512,512]
    output_dim = dim
    # dataset
    base_ds = Checkerboard_Dataset(
        num_rows=4,
        dataset_size= 50000
    )
    lie_ds = TorusLieWrapper(base_ds)
    angle_ds = AngleTorusWrapper(lie_ds)  # each item: (num_points, 2) in [-pi, pi)
    loader = DataLoader(
        angle_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_base_ds = Checkerboard_Dataset(
        num_rows=4,
        dataset_size=4096
    )
    val_lie_ds = TorusLieWrapper(val_base_ds)
    val_angle_ds = AngleTorusWrapper(val_lie_ds)
    val_loader = DataLoader(
        val_angle_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    # diffusion + score model
    diffusion = TDMDiffusion(dim=dim, integrator_type="Euler", simplified_param=True).to(device)
    model = TDM_SimpleScoreMLP(
        dim=dim,
        x_lifting_dim=x_lifting_dim,
        time_embedding_half_dim=time_embedding_half_dim,
        time_embedding_scale=time_embedding_scale,
        position_fourier_bands=position_fourier_bands,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        with_sincos_position=True,
        only_sincos_position=True
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
        for batch_idx, f0 in enumerate(pbar, start=1):
            # f0: (B, 2)
            f0 = f0.to(device, dtype=torch.float32)
            # sample noised state + target score
            # latents = (v_t, f_t), each (B, 2)
            # t_scalar: (B, 1)
            (v_t, f_t), target_score, t_scalar = diffusion.sample_forward(
                f0=f0,
                total_time=total_time,
                t_dist_kw=t_dist_kw,
                v0_dist_kw="zero",
                return_time=True,
            )
            pred_score = model(f_t, v_t, t_scalar)  # (B, 2)
            if use_weighted_loss:
                loss = weighted_score_loss(pred_score, target_score, t_scalar, total_time)
            else:
                loss = diffusion.loss_diffusion(pred_score, target_score, t_scalar)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            diagnostics = score_diagnostics(pred_score.detach(), target_score.detach(), t_scalar, total_time)
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
    plt.savefig("training_loss.png", dpi=150)
    plt.show()
    # save model
    torch.save(model.state_dict(), "simple_score_mlp_checkerboard.pt")
    print("Training done. Saved model to simple_score_mlp_checkerboard.pt")
if __name__ == "__main__":
    main()