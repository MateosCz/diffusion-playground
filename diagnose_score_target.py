import torch

from src.data import wrap_angle, wrap_pos
from src.distribution import WrappedNormalDistribution, sigma_norm


def wrapped_normal_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    trunc_n: int,
    period: float = 2 * torch.pi,
) -> torch.Tensor:
    density = torch.zeros_like(x)
    for k in range(-trunc_n, trunc_n + 1):
        density = density + torch.exp(-((x - mu + k * period) ** 2) / (2 * sigma**2))
    return torch.log(density)


def finite_difference_score(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    trunc_n: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    delta = torch.full_like(x, eps)
    return (
        wrapped_normal_log_prob(x + delta, mu, sigma, trunc_n)
        - wrapped_normal_log_prob(x - delta, mu, sigma, trunc_n)
    ) / (2 * delta)


def main():
    trunc_n = 25
    mu = torch.tensor([[0.0]])
    sigma = torch.tensor([[0.5]])
    x = torch.tensor([[0.4]])

    wrapped_normal = WrappedNormalDistribution(mu, sigma, trunc_n)
    analytic_score = wrapped_normal.score(x)
    fd_score = finite_difference_score(x, mu, sigma, trunc_n)

    samples = sigma * torch.randn(100_000, 1)

    print(f"analytic score:        {analytic_score.item(): .6f}")
    print(f"finite-diff score:     {fd_score.item(): .6f}")
    print(f"absolute error:        {(analytic_score - fd_score).abs().item(): .6e}")
    print(f"raw sample range:      [{samples.min().item(): .3f}, {samples.max().item(): .3f}]")
    print(
        "wrap_pos sample range: "
        f"[{wrap_pos(samples, x_range=2 * torch.pi).min().item(): .3f}, "
        f"{wrap_pos(samples, x_range=2 * torch.pi).max().item(): .3f}]"
    )
    print(
        "wrap_angle range:      "
        f"[{wrap_angle(samples).min().item(): .3f}, {wrap_angle(samples).max().item(): .3f}]"
    )
    print(f"sigma_norm:            {sigma_norm(sigma.flatten(), N=trunc_n, sn=2_000).tolist()}")


if __name__ == "__main__":
    main()
