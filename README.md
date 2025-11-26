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
<img width="772" height="502" alt="image" src="https://github.com/user-attachments/assets/e1ba32bd-ccaf-45fd-adef-21dfe92993cd" />

1. Requirements

Install dependencies:
Ensure:
pip install numpy pandas pillow torch torchvision scikit-learn matplotlib seaborn tqdm
Python ≥ 3.8

PyTorch ≥ 1.13

CUDA-enabled GPU recommended (but CPU also supported)
2. Data Organization

Place your dataset in the following structure:
<img width="630" height="334" alt="image" src="https://github.com/user-attachments/assets/1a20f5f3-1961-45e7-b1c4-575ca16dbdb0" />

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
