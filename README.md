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
x_0 = flow.sample_prior(x_data)
t, x_t, x_T = flow.sample_training_pair(x_data, x_0)
prediction = model(t, x_t)
loss = flow.loss(prediction, x_T, t=t)

# The learned ODE also calls the model as model(t, x).
times, trajectory = flow.sample(
    model,
    x_0,
    n_steps=100,
    return_trajectory=True,
)
```

RFM models predict velocity directly. RG-VFM models predict the terminal state
``x_T``, which the method converts to a velocity at the current state ``x_t``
before ODE integration.

For the 2D flat-torus checkerboard, the velocity-regression experiment can be
started with:

```bash
python -m src.litTrain.trainLitRFMMLP
```

Both the RFM and RG-VFM training entry points periodically generate validation
samples and save a distribution-selected checkpoint using checkerboard
histogram TV. RG-VFM additionally logs its gain over the identity endpoint
baseline; RFM logs its gain over the zero-velocity baseline. These diagnostics
should be used alongside regression loss because either loss contains a large
irreducible component that need not correlate with sample quality.

Evaluate the best distribution-selected checkpoint without relying on a
hard-coded notebook architecture:

```bash
python -m src.litTrain.evalFlatTorus2D --method rfm
python -m src.litTrain.evalFlatTorus2D --method rgvfm
```

RG-VFM also supports an extrinsic ambient-space parameterization.  The
manifold owns the embedding metadata and conversions; for example a
``FlatTorus01(dim=d)`` is embedded in ``R^(2d)`` with cosine/sine pairs:

```python
flow = RGVFM(FlatTorus01(dim=2), support="extrinsic")
x_0 = flow.sample_prior(x_data)  # shape: (batch, 4), Gaussian in R^4
t, x_t, x_T = flow.sample_training_pair(x_data, x_0)

ambient_sample = flow.sample(model, x_0, n_steps=100)
intrinsic_sample = flow.to_intrinsic(ambient_sample)  # shape: (batch, 2)
```
