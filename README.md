# Implementing_DPO_PPO
DPO (Direct Preference Optimization) Implements Direct Preference Optimization, aligning language models using pairwise preference data without reward models or PPO.  PPO (Proximal Policy Optimization) Implements Proximal Policy Optimization for RLHF, fine-tuning language models with a reward model and KL-constrained updates.

## Purpose

This project focuses on understanding RLHF by implementing each component explicitly, without relying on high-level alignment frameworks. The emphasis is on clarity, correctness, and learning how modern LLM alignment works end to end.

# Direct Preference Optimization (DPO) from Scratch

This repository contains a **from-scratch implementation of Direct Preference Optimization (DPO)** under strict compute and memory constraints, designed for **mechanistic analysis of preference optimization** rather than large-scale training.

The implementation avoids RL rollouts and reward model training, and instead directly optimizes a closed-form objective derived from KL-regularized RLHF.

---

## Theory: Direct Preference Optimization

### Problem setup
We consider a dataset of human preference pairs. For each prompt \( x \), we observe two responses:
- \( y^+ \): preferred (chosen)
- \( y^- \): less preferred (rejected)

Our goal is to train a policy model \( \pi_\theta(y \mid x) \) such that:
\[
\pi_\theta(y^+ \mid x) > \pi_\theta(y^- \mid x)
\]

while remaining close to a reference policy \( \pi_{\text{ref}} \).

---

### From RLHF to DPO
Standard RLHF optimizes:
\[
\max_{\pi} \; \mathbb{E}_{y \sim \pi(\cdot \mid x)}[r(x,y)] - \beta \, \mathrm{KL}(\pi \| \pi_{\text{ref}})
\]

where:
- \( r(x,y) \) is a learned reward model
- \( \beta \) controls deviation from the reference

This requires:
1. training a reward model
2. sampling rollouts
3. running policy optimization (e.g., PPO)

DPO eliminates these steps by **deriving a direct objective from preference data**.

---

### DPO objective
Let:
\[
\Delta_\pi = \log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x)
\]
\[
\Delta_{\text{ref}} = \log \pi_{\text{ref}}(y^+ \mid x) - \log \pi_{\text{ref}}(y^- \mid x)
\]

DPO minimizes the loss:
\[
\mathcal{L}_{\text{DPO}} =
- \log \sigma\left(\beta \left( \Delta_\pi - \Delta_{\text{ref}} \right)\right)
\]

where \( \sigma \) is the logistic sigmoid.

---

### Intuition
- The model is rewarded when it prefers \( y^+ \) over \( y^- \) **more strongly than the reference model does**
- The reference subtraction acts as an implicit KL regularizer
- No reward model or RL loop is required

The hyperparameter \( \beta \) controls how aggressively the policy deviates from the reference.

---

### Length normalization
Token log-probabilities are averaged over response length:
\[
\log \pi(y \mid x) = \frac{1}{|y|} \sum_{t} \log \pi(y_t \mid x, y_{<t})
\]

This mitigates length bias, which otherwise favors shorter responses in pairwise comparisons.

---

## Dataset

### Anthropic HH-RLHF
We use a subset of the **Anthropic Helpful–Harmless RLHF** dataset.

Each example contains:
- `chosen`: a full conversation transcript ending in a preferred assistant response
- `rejected`: the same conversation context ending in a less preferred response

There is **no explicit prompt field**.

---

### Prompt extraction
For each pair:
- The prompt is defined as the shared conversation prefix up to the final `"Assistant:"` marker
- The response is the assistant’s final turn

This ensures that chosen and rejected responses are compared under **identical context**, as required by DPO.

---

### Subsampling
Due to compute constraints, we train on a small random subset (e.g., 200–300 preference pairs).  
This is sufficient to study optimization behavior and stability without aiming for convergence.

---

## Models

### Policy model
- **Base model**: `Qwen/Qwen2.5-0.5B`
- **Training**: LoRA adapters (rank 8) applied to attention projections
- **Precision**: bfloat16
- **Gradient checkpointing** enabled for memory efficiency

Only the LoRA parameters are updated during training.

---

### Reference model
- Same base architecture and weights as the policy
- Loaded in **4-bit quantized precision**
- Fully frozen (no gradient updates)

The reference provides a stable anchor and prevents uncontrolled policy drift.

---

## Implementation Details

### Log-probability computation
- Prompt tokens are masked out (`-100`) so only response tokens contribute to the loss
- Log-probabilities are computed via token-level cross-entropy
- Sequence-level scores are length-normalized

---

### Optimization
- Batch size = 1 with gradient accumulation
- AdamW optimizer
- Gradient clipping to stabilize updates
- Short-horizon training (few hundred steps)

The goal is to analyze **optimization dynamics**, not to achieve peak performance.

---

## Scope and Intended Use

This repository is intended for:
- understanding DPO mechanics
- analyzing credit assignment and stability
- comparing preference optimization methods under controlled conditions

It is **not** a production-scale training setup.

---

## Future Extensions
- PPO and GRPO implementations under the same pipeline
- KL-drift and preference margin diagnostics
- Shortcut sensitivity tests (length, refusal, hedging bias)
- Out-of-distribution preference evaluation

  # RLHF from Scratch: SFT, Reward Modeling, and Policy Optimization

This project implements the full Reinforcement Learning from Human Feedback (RLHF) pipeline from scratch. It covers supervised fine-tuning, reward model training, and policy optimization to align language models using human preferences.

---



Each notebook represents a distinct stage of the RLHF pipeline.

---

## 1. Supervised Fine-Tuning (SFT)

**File:** `SFT.ipynb`

Implements supervised fine-tuning of a pretrained language model using instruction–response pairs. This produces an instruction-following policy that serves as the base model for alignment.

**Key concepts**
- Instruction tuning  
- Causal language modeling  
- Prompt–response masking  

---

## 2. Reward Model Training

**File:** `RM Training.ipynb`

Trains a reward model using pairwise preference data (chosen vs. rejected responses). The reward model learns to score outputs according to human preferences.

**Key concepts**
- Pairwise preference learning  
- Binary preference loss  
- Reward modeling  

---

## 3. Reinforcement Learning from Human Feedback (RLHF)

**File:** `RLHF.ipynb`

Aligns the policy using preference signals through policy optimization methods such as PPO or DPO-style objectives. The policy is optimized to maximize reward while remaining close to a reference model.

**Key concepts**
- Policy optimization  
- KL regularization  
- Alignment stability  

---
