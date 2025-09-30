# Neural‑Network State‑Space Estimators

This repository provides all code and data to reproduce the simulations and experiments from the paper:

> **Neural‑Network State‑Space Estimators**
> Minxing Sun, Li Miao, Qingyu Shen, Yao Mao\*, Qiliang Bao

---

## 📁 Directory Structure

```
Matlab/                  # Main MATLAB scripts
  ├─ Test.m              # 1. PROBLEM STATEMENT validation
  ├─ NNSSE_Sin.m         # 2. SIMULATION on noisy sine wave
  ├─ NNSSE_DR.m          # 3. EXPERIMENT 1: dual‑reflection mirror
  ├─ NNSSE_Drone.m       # 4. EXPERIMENT 2: UAV trajectory
  ├─ StateSpaceModel_*.m # NNSSM variants
  └─ Estimator_*.m       # EKF / UKF / PF implementations
  └─ NNSSE_*.mat         # All data to reproduce result in paper

Python/                  # Neural‑network training scripts
  ├─ 01_RNN_train_2025.py# Format: <id>_<net>_<task>_<date>.py
  ├─ …                   # Other network types
  ├─ config.json         # Paths & hyper‑parameters
  └─ NN_*.bat            # One‑click training on Windows

EstimationResult/        # Prediction results for MATLAB comparisons
  ├─ Data_1  # limited‑training nets (Exp‑1)
  ├─ Data_2  # full‑training    nets (Exp‑1)
  ├─ Data_3  # limited‑training nets (Exp‑2)
  ├─ Data_4  # full‑training    nets (Exp‑2)
  └─ Data_5  # NNSSEs for Sin   nets (Sim-0)
```

---

## 1. PROBLEM STATEMENT Validation (`Matlab/Test.m`)

Reproduces Fig. 2. Compares classical Kalman filters on states `[p_i]`, `[p_i,v_i]`, `[p_i,v_i,a_i]`, `[p_i,v_i,a_i,ẍ_i]` against position-stack estimators (E2P, E3P, E4P, E4PVW, E4PRW, E4PTRW).  Gaussian noise is regenerated each run, so individual results vary slightly.

---

## 2. SIMULATION (`Matlab/NNSSE_Sin.m`)

Emulates a manoeuvring target with:

- **Ground truth**: 1 Hz sine, amplitude 10
- **Sampling**: 200 Hz, latency 0.15 s → 3‑frame prediction
- **Noise**: zero‑mean Gaussian, unit covariance

The script picks a `StateSpaceModel_*.m` and an `Estimator_*.m` to assemble each NNSSE variant.

---

## 3. EXPERIMENT 1: Dual‑Reflection Mirror (`Matlab/NNSSE_DR.m`)

Bench setup: laser, CCD camera, fast steering mirror, validation mirror.  Camera at 200 Hz with 0.015 s latency → 3‑frame prediction.  Baseline NN predictions are loaded from:

```
EstimationResult/Data_1  # limited‑training
EstimationResult/Data_2  # full‑training
```

---

## 4. EXPERIMENT 2: UAV Trajectory (`Matlab/NNSSE_Drone.m`)

Yaw data from a real UAV, including zenith‑crossing reversals.  Scripts load neural baselines from:

```
EstimationResult/Data_3  # limited‑training
EstimationResult/Data_4  # full‑training
```

---

## 5. Neural‑Network Training (`Python/`)

1. Edit `config.json` to set paths and parameters (train/test/estimate).
2. Run training scripts, e.g.:

   ```bash
   cd Python
   python 00011_RNN_Train_20241129.py
   ```

   or on Windows:

   ```bat
   NN_FullTrain.bat
   ```
3. Copy generated files into `Matlab/EstimationResult/Data_X` for MATLAB.

---

Contact: sunminxing20@gmail.com

© 2025 Minxing Sun et al. Licensed under MIT.
