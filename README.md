# Modulation-Based Spectrogram Feature Extraction for Sleep Stage Classification
This repository contains the source code used in the paper:

“Modulation-Based Feature Extraction for Robust Sleep Stage Classification Across Apnea-Based Cohorts.”

A deep-learning framework for automated sleep stage classification using modulation spectrograms derived from EEG signals, evaluated across AHI-based cohorts for robust performance and generalizability.
<img width="993" height="450" alt="image" src="https://github.com/user-attachments/assets/14f24c70-7767-4fb4-a611-2dde5bd40958" />
The repo provides:

Python code for Modulation Spectrogram generation

MATLAB code for STFT and CWT spectrogram generation

Training and evaluation pipelines for:

The Modulation + Sequence Model (EEGSNet-like)

The CNN-only baseline

Utilities for dataset organization, sequence construction, and cross-validation

Example structure and instructions for reproducing all experiments

## Repository Structure

│
├── data/
│ └── (optional placeholder for datasets)
│
├── spectrogram_generation/
│ ├── modulation_python/
│ │ └── modulation_spectrogram.py # Python Modulation Spectrogram generation
│ │
│ ├── STFT_MATLAB/
│ │ └── stft_generation.m # MATLAB STFT (EEGSNet-style)
│ │
│ └── CWT_MATLAB/
│ └── cwt_generation.m # MATLAB CWT (76×60 output)
│
├── src/
│ ├── modulation_core.py # Dataset loader, utils, model blocks
│ ├── train_eval_modulation.py # Main Modulation (EEGSNet-like) training + evaluation
│ └── train_cnn_baseline.py # CNN-only baseline model
│
├── results/
│ ├── README.md # Empty placeholder (results auto-generated)
│
└── README.md # Project documentation
1. Requirements

Install dependencies:
Ensure:
pip install numpy pandas pillow torch torchvision scikit-learn matplotlib seaborn tqdm
Python ≥ 3.8

PyTorch ≥ 1.13

CUDA-enabled GPU recommended (but CPU also supported)
2. Data Organization

Place your dataset in the following structure:
DATA_ROOT/
│
├── Training_Set/
│   └── Subject folders (S001, S002, ...)
├── Validation_Set/
├── Testing_Set/
│
Each subject folder:
│
└── {W, N1, N2, N3, R}/
       └── C4-M1_XXXX.png
This structure matches the folder layout used in the experiments.
3. Spectrogram Generation
Modulation Spectrogram (Python)
STFT & CWT Spectrograms (MATLAB)
4. Training the Models
A) Modulation Spectrogram + Sequence Model (EEGSNet-like)+ change the dataset and image size, and train using STFT and CWT spectrograms as well.
B) CNN-Only Baseline
6. Citation

If you use this code, please cite.
7. License

This code is released for academic research purposes only.
