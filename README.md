# diffusion-playground

Experiments with diffusion and flow-matching models on periodic and material
data.

## Source layout

The two model families intentionally use different conventions:

```text
src/
  manifolds/       geometry, geodesics, tangent operations and priors
  flow_matching/   deterministic probability paths and FM objectives
  ode/             generic Euler, Heun and RK4 integration
  diffusion/       stochastic forward/reverse diffusion processes
```

Diffusion classes retain the repository's existing ``sample_forward`` /
``sample_backward`` convention.  Flow matching uses the common time-first ODE
convention:

```python
from src.flow_matching import RGVFM
from src.manifolds import FlatTorus01

flow = RGVFM(FlatTorus01(dim=2))
t, x_t, endpoint = flow.sample_training_pair(x_data)
prediction = model(t, x_t)
loss = flow.loss(prediction, endpoint)

# The learned ODE also calls the model as model(t, x).
times, trajectory = flow.sample(
    model,
    x_base,
    n_steps=100,
    return_trajectory=True,
)
```

RFM models predict velocity directly.  RG-VFM models predict a path endpoint,
which the method converts to a velocity before ODE integration.

RG-VFM also supports an extrinsic ambient-space parameterization.  The
manifold owns the embedding metadata and conversions; for example a
``FlatTorus01(dim=d)`` is embedded in ``R^(2d)`` with cosine/sine pairs:

```python
flow = RGVFM(FlatTorus01(dim=2), support="extrinsic")
x_base = flow.sample_prior(x_data)  # shape: (batch, 4), Gaussian in R^4
t, x_t, endpoint = flow.sample_training_pair(x_data, x_base)

ambient_sample = flow.sample(model, x_base, n_steps=100)
intrinsic_sample = flow.to_intrinsic(ambient_sample)  # shape: (batch, 2)
```
