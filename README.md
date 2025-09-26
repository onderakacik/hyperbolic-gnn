# Hyperbolic Graph Neural Networks (HGNN)

Reproduction and extension of Hyperbolic GNNs with Euclidean, Poincaré, and Lorentz manifolds. Adds a simpler prediction head via tangent-space mapping (log-map at the origin + linear layer). Includes training/evaluation for graph prediction (ZINC) and node classification (AIRPORT).

## Installation

```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For GPU, install a CUDA-compatible torch build per https://pytorch.org/get-started/locally/
```

## Data preparation

- ZINC (creates JSON files under `data/zinc/`)
```bash
cd data/zinc
python get_data.py   # downloads CSV and generates molecules_{train,valid,test}_zinc.json
cd ../../
```

- AIRPORT
  - Dataset files are included under `data/airport/` (`airport.p`, etc.). No action needed.

## Using main.py

Common flags:
- `--task`: `zinc` | `airport`
- `--select_manifold`: `euclidean` | `poincare` | `lorentz`
- `--use_tangent`: enable tangent-space prediction head
- `--debug`: use ~10% of the dataset for quicker runs
- `--seed`: random seed
- Training controls: `--max_epochs`, `--patience`
- ZINC-specific: `--prop_idx` in {0,1,2}. Based on `data/zinc/get_data.py`: 0=QED, 1=logP, 2=SAS

### ZINC: Graph-level regression

Baseline (centroid-based):
```bash
# Euclidean, logP
python main.py --task zinc --select_manifold euclidean --prop_idx 1 --max_epochs 20 --patience 5

# Poincaré, QED
python main.py --task zinc --select_manifold poincare --prop_idx 0 --max_epochs 20 --patience 5

# Lorentz, SAS
python main.py --task zinc --select_manifold lorentz --prop_idx 2 --max_epochs 20 --patience 5
```

Tangent-space extension:
```bash
# Poincaré + tangent, QED
python main.py --task zinc --select_manifold poincare --use_tangent --prop_idx 0 --max_epochs 20 --patience 5

# Lorentz + tangent, logP
python main.py --task zinc --select_manifold lorentz --use_tangent --prop_idx 1 --max_epochs 20 --patience 5

# Euclidean + tangent, SAS
python main.py --task zinc --select_manifold euclidean --use_tangent --prop_idx 2 --max_epochs 20 --patience 5
```

Quick debug runs (10% data):
```bash
python main.py --task zinc --debug --select_manifold poincare --prop_idx 1 --max_epochs 5 --patience 2
```

### AIRPORT: Node classification

Baseline (centroid-based):
```bash
python main.py --task airport --select_manifold poincare --max_epochs 15000 --patience 500
python main.py --task airport --select_manifold lorentz  --max_epochs 15000 --patience 500
python main.py --task airport --select_manifold euclidean --max_epochs 15000 --patience 500
```

Tangent-space extension:
```bash
python main.py --task airport --select_manifold poincare --use_tangent --max_epochs 15000 --patience 500
python main.py --task airport --select_manifold lorentz  --use_tangent --max_epochs 15000 --patience 500
python main.py --task airport --select_manifold euclidean --use_tangent --max_epochs 15000 --patience 500
```

### Logging and reproducibility
- Logs: `log/<run_name>.log`
- Set `--seed` for reproducibility. The script sets deterministic cuDNN flags.

### SLURM examples
Job scripts are provided in `job_files/` (e.g., `zinc_poincare.job`, `airport.job`) showing `srun` invocations with typical arguments.

## Project layout (key folders)
- `manifolds/`: Euclidean, Poincaré, Lorentz implementations
- `models/`: `RiemannianGNN`, task heads
- `tasks/`: `GraphPredictionTask`, `NodeClassificationTask`
- `params/`: task/manifold-specific CLI parameters
- `data/`: dataset assets and preprocessors

## Reference
- Liu, Nickel, Kiela. Hyperbolic Graph Neural Networks (2019).
