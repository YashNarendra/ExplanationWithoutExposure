# Explanation Without Exposure: Security-Aware Explanations via Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
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
git clone https://github.com/YourRepo/ExplanationWithoutExposure.git
cd ExplanationWithoutExposure

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Ensure your `requirements.txt` contains the following versions to match our experiments:

```text
torch>=2.1.0
transformers>=4.37.0
accelerate
peft
bitsandbytes
pillow
scikit-learn
```

**System Requirements:**
* **Python:** 3.10+
* **GPU:** CUDA-capable GPU (Recommended: 16GB VRAM for training, 6GB for inference)

## 🚀 Usage

### 1. Training

To fine-tune the SLM on your own dataset of (Plot, Explanation) pairs, configure the parameters in `train.py` and run:

```bash
python train.py
```

*Note: The default script uses dummy data. Replace `dummy_data` in `train.py` with your actual JSON dataset loader.*

### 2. Inference

To generate an explanation for a specific SHAP or LIME plot:

```python
from model import ExplanationVLM
from PIL import Image
import torch

# Load Model
model = ExplanationVLM()
model.slm.load_adapter("saved_models/slm_lora", adapter_name="default")
model.projection.load_state_dict(torch.load("saved_models/projector.pt"))

# Run Inference
image = Image.open("examples/sample_shap_plot.png")
prompt = "Explain the global feature importance shown in this plot."
explanation = model.generate(image, prompt)

print(explanation)
```

## 📂 Project Structure

```text
.
├── dataset.py       # Custom PyTorch Dataset for loading Image-Text pairs
├── model.py         # Definition of the VLM (SigLIP + Projector + Phi-2)
├── train.py         # QLoRA fine-tuning loop
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ✏️ Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{anonymous2024explanation,
  title={Explanation Without Exposure: Security-Aware Explanations via Small Language Models},
  author={Anonymous Authors},
  journal={Under Review at ACL},
  year={2024}
}
```
