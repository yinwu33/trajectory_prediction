# Trajectory Prediction

A PyTorch Lightning-based framework for motion prediction using the Argoverse 2 dataset, implementing VectorNet and SIMPL models.

## Environment Setup

```bash
# Create/update the project environment from pyproject.toml
uv sync
```

## Data Preparation

Download the [Argoverse 2 Motion Forecasting dataset](https://www.argoverse.org/av2.html) and place it under `./data/`:

```
data/
├── train/
├── val/
└── test/
```

The default data root can be overridden via config (see `configs/datamodule/av2_base.yaml`).

## Run Training

Train with the default VectorNet config:

```bash
uv run main.py
```

Train with SIMPL:

```bash
uv run main.py --config-name config_simpl.yaml
```

Override config options via command line (Hydra syntax):

```bash
# Change experiment name and batch size
uv run main.py exp_name=my_exp datamodule.batch_size=128

# Resume from checkpoint
uv run main.py resume_from=/path/to/checkpoint.ckpt

# Run a named experiment preset
uv run main.py +experiment=vectornet_large
```

## Models

### VectorNet

- **Paper**: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)
- **Config**: `configs/config_vectornet.yaml`
- Agent-centric preprocessing; polylines encoded via local graph, fused via global graph transformer
- Multi-modal prediction with `k=6` trajectories, 60-step (6s) future horizon

### SIMPL

- **Reference**: [SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline](https://github.com/HKUST-Aerial-Robotics/SIMPL)
- **Config**: `configs/config_simpl.yaml`
- Scene-centric preprocessing; actor and lane features fused via relative positional encoding (RPE) transformer
- Parametric trajectory output (Bezier / monomial curve fitting or direct waypoints)

## Project Structure

```
trajectory_prediction/
├── main.py                     # Training entry point (Hydra + Lightning)
├── configs/                    # Hydra config files
│   ├── config_vectornet.yaml
│   ├── config_simpl.yaml
│   ├── datamodule/
│   ├── model/
│   ├── trainer/
│   └── experiment/
├── models/
│   ├── vectornet/
│   ├── simpl/
│   └── metrics.py              # ADE, FDE, minADE, minFDE
├── datamodule/
│   ├── av2_vectornet.py
│   ├── av2_simpl.py
│   └── datasets/av2/
├── callbacks/
│   └── viz.py                  # Trajectory visualization callback
└── utils/
    ├── viz.py / viz_av2.py
    ├── numpy.py
    └── init_weights.py
```

## Metrics

| Metric | Description |
|--------|-------------|
| ADE    | Average Displacement Error (all timesteps, best mode) |
| FDE    | Final Displacement Error (last timestep, best mode) |
| minADE | minADE over k=6 predicted modes |
| minFDE | minFDE over k=6 predicted modes |

## Model Zoo and Benchmark

[Benchmark](./benchmark.md)

## Acknowledgements

- [VectorNet](https://arxiv.org/abs/2005.04259) — Gao et al., CVPR 2020
- [SIMPL](https://github.com/HKUST-Aerial-Robotics/SIMPL) — HKUST Aerial Robotics Group
- [Argoverse 2](https://www.argoverse.org/av2.html) — Argoverse 2 Motion Forecasting dataset
