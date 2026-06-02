# Adversarial Networks

# Basic GANs — Vanilla GAN vs. LSGAN on FashionMNIST

Implements and compares a Vanilla GAN and a Least Squares GAN (LSGAN) for unconditional image generation on FashionMNIST. Demonstrates how replacing Binary Cross-Entropy with MSE loss stabilizes adversarial training and reduces generator loss.

---

## Dataset

- **Source:** FashionMNIST (auto-downloaded via `torchvision`)
- **Train:** 60,000 grayscale 28×28 images across 10 clothing categories
- **Preprocessing:** Normalized to [−1, 1] (mean=0.5, std=0.5)

---

## Architecture

Both models use fully-connected (MLP) architectures with shared hyperparameters:

| Hyperparameter | Value |
|---------------|-------|
| Input (latent) size | 64 |
| Hidden size | 256 |
| Output (image) size | 784 (28×28 flattened) |

**Discriminator:**
```
Linear(784→256) → LeakyReLU(0.2) → Linear(256→256) → LeakyReLU(0.2) → Linear(256→1) [→ Sigmoid for GAN only]
```

**Generator:**
```
Linear(64→256) → ReLU → Linear(256→256) → ReLU → Linear(256→784) → Tanh
```

**Key difference:** Vanilla GAN uses BCELoss (Sigmoid output); LSGAN removes Sigmoid and uses MSELoss on raw logits.

---

## Training

- **Optimizer:** Adam, lr=0.0001 (reduced from 0.0002 to prevent non-convergence)
- **Batch size:** 256
- **Total training:** 1,000 epochs (2 runs of 500 with checkpoint-based resumption)
- **Hardware:** Apple MPS (Metal Performance Shaders)

---

## Results (Epoch 1000)

| Metric | Vanilla GAN | LSGAN |
|--------|-------------|-------|
| Discriminator Loss | ~0.978 | ~0.287 |
| Generator Loss | ~1.401 | **~0.570** |
| D Accuracy (Real) | 73.2% | **77.5%** |
| D Accuracy (Fake) | 20.1% | 15.0% |

> LSGAN losses are lower in absolute scale because MSE operates on a different range than BCE.

---

## Key Findings

- LSGAN generator loss was **2.4× lower** than vanilla GAN (0.570 vs. 1.401), confirming more stable adversarial training
- LSGAN achieved higher discriminator accuracy on real images (77.5% vs. 73.2%), indicating better class separation
- Vanilla GAN showed a sharp instability spike around epoch 750 where generator loss jumped and partially recovered — LSGAN remained stable across all 1,000 epochs
- Qualitatively, vanilla GAN produced sharper-looking images at epoch 1,000 while LSGAN appeared to have plateaued earlier

---

## Setup

```bash
pip install torch torchvision numpy matplotlib
```

Open `adversarial-networks/GAN_LSGAN.ipynb` in Jupyter. FashionMNIST downloads automatically on first run. Pretrained weights are available in `weights/`.

