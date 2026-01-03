# Explanation Without Exposure: Security-Aware Explanations via Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)

**Anonymous Repository for ACL Submission**

This repository contains the official PyTorch implementation of the paper **"Explanation Without Exposure: Security-Aware Explanations via Small Language Models"**.

## 📄 Abstract
High-stakes domains such as healthcare and finance require interpretable machine learning but are often constrained by strict data privacy regulations. Cloud-hosted Large Language Models (LLMs) pose a security risk when processing sensitive feature attribution data.

This framework introduces a privacy-preserving solution using **Small Language Models (SLMs)** deployed within a trusted local environment. By treating post-hoc explanations (SHAP/LIME plots) as visual inputs, our multimodal architecture (SigLIP + Phi-2) generates faithful natural language explanations without externalizing sensitive data.

## 🏗️ Architecture
The model follows a Vision-Language architecture optimized for edge deployment:
1.  **Vision Encoder:** SigLIP-SO400M (Frozen, 384px resolution) for high-fidelity OCR and chart reasoning.
2.  **Projection Layer:** A trainable MLP mapping visual features to the language embedding space.
3.  **Language Decoder:** Microsoft Phi-2 (2.7B), fine-tuned using QLoRA (4-bit quantization).

## 🛠️ Installation

We recommend using a virtual environment or Conda environment.

```bash
# Clone the repository
git clone [https://github.com/YourRepo/ExplanationWithoutExposure.git](https://github.com/YourRepo/ExplanationWithoutExposure.git)
cd ExplanationWithoutExposure

# Install dependencies
pip install -r requirements.txt
