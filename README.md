# Image Captioning Model (Encoder-Decoder: CNN + RNN)

This repository contains an implementation of an image captioning model that generates descriptive captions for images. The model uses a pre-trained CNN as an encoder (ResNet50) to extract image features and an RNN (LSTM) as a decoder to generate captions word-by-word.

## Overview

The model is composed of two main components:

- **Encoder (CNN):**  
  Uses a pre-trained ResNet50 with frozen weights to extract visual features. The final fully-connected layer is replaced with a custom linear embedding layer and a batch normalization layer.

- **Decoder (RNN):**  
  Uses an LSTM network to generate captions from the encoded image features. The decoder includes:
  - An embedding layer for tokenizing captions.
  - An LSTM network for sequence generation.
  - A linear layer to map LSTM outputs to a probability distribution over the vocabulary.
  - Dropout for regularization.

The complete model is wrapped in an `Encoder_Decoder` class, which handles both training and inference (sampling).

## Model Architecture

- **Encoder (EncoderCNN):**
  - **Base Model:** ResNet50 (pre-trained on ImageNet, weights frozen).
  - **Embedding Layer:** Projects features to a specified `embed_size`.
  - **Batch Normalization:** Stabilizes and accelerates training.
  
- **Decoder (DecoderRNN):**
  - **Embedding Layer:** Converts word indices to embeddings.
  - **LSTM:** Generates hidden states sequentially.
  - **Linear Layer:** Maps hidden states to vocabulary scores.
  - **Dropout:** Helps prevent overfitting.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/) (optional, for visualization)

Install the required Python packages via pip:

```bash
pip install torch torchvision nltk matplotlib
