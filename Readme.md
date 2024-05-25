# Positional Embeddings in Transformer Models

This project demonstrates the implementation of absolute and rotary positional embeddings (RoPE) in a transformer model using PyTorch. The goal is to explore different positional embeddings mechanisms used to encode positions of tokens along with the context to improve the performance of self-attention mechanisms in transformers.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [References](#references)


## Overview

Positional embeddings are crucial in transformer models as they provide information about the position of tokens in a sequence. This project covers:

1. **Absolute Positional Embeddings**: Traditional method where positional information is added directly to the token embeddings.
2. **Rotary Positional Embeddings (RoPE)**: A method where positional information is applied to the query and key vectors, enhancing the attention mechanism's ability to capture relative positions.

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/positional-embeddings-transformer.git
    cd positional-embeddings-transformer
    ```

2. Install the required packages:
    ```bash
    pip install torch numpy
    ```

## Refrences

