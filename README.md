# Autoencoder Image Denoiser

Este projeto implementa um **Autoencoder** para remoção de ruído em imagens utilizando **PyTorch**. O modelo é treinado para remover **ruído gaussiano** adicionado às imagens, restaurando a versão limpa da imagem de entrada.

This projecrt implements an **Autoencoder** for image denoising by using **Pytorch**. The model is trained to remove **Gaussian Noise** that was previously added to the images, restoring the clean version.

## Sumary

- [Introduction](#introduction)
- [Autoencoder's Architecture](#arch)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Introduction

"Denoiser" is a fundamental task in image processing, specially in areas as computer vision and data analysis. This project uses an **Autoencoder**, a Neural Network that learns to represent the data effectively to remove noise added to images.

## Autoencoder's Architecture

The model consists in two parts mainly:
- **Encoder:** Reduces the dimensions of the images and learns a compact representation.
- **Decoder:** . Rebuilds the images from that previous compacted representation.

### Requirements

- Python 3.10+
- PyTorch 1.7+
- Library to visualization (optional): Matplotlib

