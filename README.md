Real-Time Low-Light Image and Video Enhancement
Lightweight Retinex-Based Network (KinD++ + MobileNet Optimization)

Master’s Thesis – Computer Science (Data Science)
Leiden University – LIACS
Author: Yağmur Doğan
Supervisors: Dr. Hazel R. Doughty, Dr. Rita Pucci
Year: 2025

## Overview

Low-light conditions significantly degrade image and video quality, affecting downstream computer vision tasks such as object detection, tracking, and autonomous driving.

This repository contains the official implementation of my Master’s thesis:

“Real-Time Low-Light Image and Video Enhancement with Lightweight Retinex-based Network”

The project proposes a computationally efficient Retinex-based deep learning model optimized for real-time video enhancement, achieving:

< 35K trainable parameters

80 FPS on GPU

Competitive PSNR / SSIM / LPIPS performance

Support for images, videos, and RTSP streams

The model is based on KinD++ and enhanced with MobileNet-style depthwise separable convolutions and width multiplier scaling.

## Key Contributions

Lightweight Retinex-based architecture

Depthwise separable convolutions for reduced computation

Width multiplier (0.5) for model scaling

Mixed precision training

Multi-threaded video inference pipeline

Real-time FPS monitoring

GAN vs Non-GAN ablation study

Support for continuous video streams (RTSP compatible)

Method Overview

The model follows the Retinex formulation:

I=R⊗L

Where:

R = Reflectance (texture, structure)

L = Illumination (lighting)

## Architecture

Decomposition Network

Separates input into illumination and reflectance

Reflectance Enhancement Branch

Preserves detail

Suppresses noise

Illumination Enhancement Branch

Adjusts brightness

Structure-aware smoothing

Reconstruction

Element-wise multiplication

Enhancements:

Depthwise separable convolutions

Channel width multiplier

Mixed precision training

Optimized GPU batch inference

## Repository Structure
├── models/                # Model architectures

├── losses/                # Loss functions (MI Loss, GAN Loss, etc.)

├── datasets/              # Dataset loaders

├── training/              # Training scripts

├── inference/             # Image, video, RTSP inference scripts

├── utils/                 # Helper utilities

├── checkpoints/           # Saved models

├── experiments/           # Evaluation scripts


## Main dependencies:

PyTorch

torchvision

OpenCV

numpy

tqdm

scikit-image

lpips

Datasets

The model was trained on:

LOL Dataset

LOLI-Street Dataset

Expected dataset structure:

data/
├── train/

│   ├── low/

│   └── high/

├── val/

│   ├── low/

│   └── high/

## Training
Train without GAN (Recommended)
python train.py --config configs/train_no_gan.yaml

Train with GAN
python train.py --config configs/train_gan.yaml


Mixed precision training is enabled by default.

## Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth


## Metrics computed:

PSNR

SSIM

LPIPS

DeltaE

## Inference
Image Inference
python infer_image.py --input path/to/image.jpg

Video Inference
python infer_video.py --input path/to/video.mp4

RTSP Stream Inference
python infer_rtsp.py --url rtsp://your_stream_url


## Features:

Multi-threaded frame pipeline

Batch GPU inference

Real-time FPS monitoring

## Results
Model Variant	Params	FPS	PSNR	SSIM	LPIPS
GAN Version	~35K	~70	Lower	Lower	Worse
MI Loss (No GAN)	~35K	>80	Higher	Higher	Better

## Key finding:

Non-GAN training with Mutual Input Loss outperforms GAN-based variants across most quantitative metrics.

## Hardware Used

NVIDIA GPU

ALICE Cluster (Leiden University)

## System Reliability & Troubleshooting

This project was developed with a strong focus on system stability, real-time performance, and robustness under varying conditions.

Key engineering practices applied:

Performance Monitoring
Continuous tracking of FPS, latency, and output consistency during inference
Real-time evaluation of system behavior under different lighting conditions and input streams
Troubleshooting & Debugging
Systematic debugging of pipeline components to identify performance bottlenecks
Isolation and testing of individual modules (preprocessing, model inference, postprocessing)
Debugging multi-threaded video pipelines to ensure stable frame processing
Root Cause Analysis
Investigation of model instability and inconsistent outputs across datasets
Analysis of failure cases (e.g., extreme low-light conditions) to improve robustness
Data-driven identification of issues related to illumination decomposition and reconstruction
Automation & Reliability
Automated training, evaluation, and benchmarking pipelines using Python
Reproducible experiment setup to ensure consistent results across runs
Logging mechanisms to track experiments and detect anomalies
System Optimization
Optimization of GPU utilization and memory usage for long-running processes
Reduction of latency through pipeline restructuring and efficient batching
Ensuring stable performance for continuous video streams (including RTSP input)
Mixed precision (FP16), Real-Time Performance, 80 FPS (GPU), Efficient memory handling,
Continuous video support

# Suitable for:
Autonomous driving
Surveillance
Night-time robotics
Real-time monitoring systems
