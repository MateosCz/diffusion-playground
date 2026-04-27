# src/train.py
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.data import Checkerboard_Dataset, TorusLieWrapper, AngleTorusWrapper
from src.scoreNN import TDM_SimpleScoreMLP
from src.diffusion import TDMDiffusion



def main():
    # -----------------------
    # Config (simple defaults)
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 512
    n_epoch = 50
    lr = 1e-3
    total_time = 2.0
    # data shape: each sample -> (num_points, dim)
    dim = 2
    # model
    x_lifting_dim = 64
    time_embedding_half_dim = 32  # must be even
    hidden_dim = [128,128]
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
    # diffusion + score model
    diffusion = TDMDiffusion(dim=dim, integrator_type="Euler", simplified_param=True).to(device)
    model = TDM_SimpleScoreMLP(
        dim=dim,
        x_lifting_dim=x_lifting_dim,
        time_embedding_half_dim=time_embedding_half_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        with_sincos_position=True,
        only_sincos_position=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # -----------------------
    # Training loop
    # -----------------------
    model.train()
    epoch_losses = []
    for epoch in range(1, n_epoch + 1):
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epoch}", leave=False)
        for batch_idx, f0 in enumerate(pbar, start=1):
            # f0: (B, num_points, 2)
            f0 = f0.to(device, dtype=torch.float32)
            # sample noised state + target score
            # latents = (v_t, f_t), each (B, num_points, 2)
            # t_scalar: (B,)
            (v_t, f_t), target_score, t_scalar = diffusion.sample_forward(
                f0=f0,
                total_time=total_time,
                t_dist_kw="uniform",
                v0_dist_kw="zero",
                return_time=True,
            )
    
            # simplest choice: use f_t as network input
            # your current model expects t.ndim==3; make (B, num_points, 1)
            # t_for_net = t_scalar[:, None, None].expand(-1, f_t.shape[1], 1)
            # t_for_score = t_scalar[:, None, None].expand(-1, f_t.shape[1], f_t.shape[2])
            # target_score = (1- torch.exp(-t_for_score))/(1 + torch.exp(-t_for_score)) * target_score - v_t/ (diffusion.sde.sigma_t(t_for_score)**2)

            pred_score = model(f_t,v_t, t_scalar)  # (B, num_points, 2)
            loss = diffusion.loss_diffusion(pred_score, target_score, t_scalar)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(
                batch=f"{batch_idx}/{len(loader)}",
                loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch:03d}/{n_epoch}]  loss={avg_loss:.6f}")
    
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