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

The training entry points default to the 1D checkerboard (`dim = 1`). Set
`dim = 2` in the training script for the 2D checkerboard. Start the
velocity-regression experiment with:

```bash
python -m src.litTrain.trainLitRFMMLP
```

Both the RFM and RG-VFM training entry points periodically generate validation
samples and save a distribution-selected checkpoint using checkerboard
histogram TV. RG-VFM additionally logs its gain over the identity endpoint
baseline; RFM logs its gain over the zero-velocity baseline. These diagnostics
should be used alongside regression loss because either loss contains a large
irreducible component that need not correlate with sample quality.

For 1D, the last section of `flat_torus_1d.ipynb` loads an explicit checkpoint
and plots learned trajectories in `(time, sin(2πx), cos(2πx))`.
For 2D, evaluate the best distribution-selected checkpoint with:

```bash
python -m src.litTrain.evalFlatTorus2D --method rfm
python -m src.litTrain.evalFlatTorus2D --method rgvfm
```

RG-VFM also supports an extrinsic ambient-space parameterization.  The
manifold owns the embedding metadata and conversions; for example a
``FlatTorus01(dim=d)`` is embedded in ``R^(2d)`` with cosine/sine pairs:

```python
from src.nn.rg_vfm_mlp import EX_RGVFMMLP

manifold = FlatTorus01(dim=2)
flow = RGVFM(manifold, support="extrinsic")
model = EX_RGVFMMLP(
    manifold,
    x_lifting_dim=256,
    time_embedding_half_dim=128,
    hidden_dim=[512, 1024, 512],
)  # raw 4D input, projected 4D endpoint, residual disabled by default
x_0 = flow.sample_prior(x_data)  # shape: (batch, 4), Gaussian in R^4
t, x_t, x_T = flow.sample_training_pair(x_data, x_0)

ambient_sample = flow.sample(model, x_0, n_steps=100)
intrinsic_sample = flow.to_intrinsic(ambient_sample)  # shape: (batch, 2)
```

In `src/litTrain/trainLitRGVFMMLP.py`, `rg_vfm_kwargs["support"]` selects
the intrinsic or extrinsic MLP automatically. Set the dataset dimension to
1 or 2; ambient network dimensions are inferred. `extrinsic_nn_kwargs` controls
the optional residual experiment independently of intrinsic settings.
Run training with `python -m src.litTrain.trainLitRGVFMMLP`.
The checkerboard callback decodes generated samples before evaluation.
`LitRGVFMMLP.sample()` continues to return model-space samples (`2 * dim` for
extrinsic); use `lit_model.rg_vfm.to_intrinsic(samples)` for plotting.
Evaluate a saved checkpoint with
`python -m src.litTrain.evalFlatTorus2D --method rgvfm --checkpoint PATH`.

## Joint CSP flow map

`Endpoint_Flowmap_CSP` uses a fixed time interval `[0, 1]` and only implements
`Xst`, `tangent_loss`, `lagrange_loss` and basic sampling from an initialized
graph. `LitCSPFlowMap` handles training pairs, the diagonal/off-diagonal split,
loss weights and prior initialization.

```python
import lightning as L
from src.csp_model import Endpoint_Flowmap_CSP
from src.flow_maps.vfm_flowmap import VFMFlowMap
from src.flow_matching import RGVFM, VFM
from src.lit.litCSPFlowMap import LitCSPFlowMap
from src.manifolds import FlatTorus01
from src.nn.ACSPVNet import Ambient_CSPVNet

# Keep MP20 positions fractional; training pairs embed them directly into R^6.
f_flow = RGVFM(FlatTorus01(dim=3), support="extrinsic")
l_flow = VFM(dim=6)
joint = Endpoint_Flowmap_CSP(VFMFlowMap(f_flow), VFMFlowMap(l_flow))
pi = Ambient_CSPVNet(f_dim=f_flow.model_dim)
model = LitCSPFlowMap(pi, joint, eta=0.5, lambda_f=1.0, lambda_l=1.0)

# graph: fractional x/pos (N, 3) in [0, 1), l (B, 6), batch (N,), atom types and edges.
# pi(graph, s, t) -> {"f": (N, 6), "l": (B, 6)} in model space.
# Training batches need at least two crystals; s/t are graph-level (B, 1).
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
generated = model.sample(graph)  # Default: one step from 0 to 1.
coordinates = f_flow.to_intrinsic(generated.x)
lattice = generated.l
```

Both losses return `{"f": loss_f, "l": loss_l}`. The default lattice uses
a Gaussian prior and Euclidean endpoint loss weighted by `(1-t)^-2`.
`VFM` in `src/flow_matching/vfm.py` implements this Euclidean branch:
`l0 ~ Normal(0, I)`, `l_t = (1-t) * l0 + t * l1`. Lattices stay in the
dataset's transformed six-dimensional parameter space without wrapping or
manifold projection. The same `VFMFlowMap` supports both `VFM` and `RGVFM`;
omitting `l_flowmap` selects `VFMFlowMap(VFM(dim=6))` automatically.
Lagrangian loss differentiates both states together and detaches only the
diagonal endpoint prediction. `ACSPVNet` accepts the graph and both times
directly, with no velocity input. Its coordinate head predicts an unprojected
endpoint using the local cosine/sine pair basis, preserving node permutation
and periodic translation equivariance. The lattice prediction is invariant
under those coordinate translations.

Graph updates only replace `x`, `pos` and `l`; the predictor computes edge
features from the current coordinates when needed. To split generated batches
with `to_data_list()`, provide templates that already have per-graph `l`
attributes (placeholder values are sufficient).
