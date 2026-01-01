# Implementing_DPO_PPO
DPO (Direct Preference Optimization) Implements Direct Preference Optimization, aligning language models using pairwise preference data without reward models or PPO.  PPO (Proximal Policy Optimization) Implements Proximal Policy Optimization for RLHF, fine-tuning language models with a reward model and KL-constrained updates.

# Direct Preference Optimization (DPO) from Scratch

This repository contains a from-scratch implementation of **Direct Preference Optimization (DPO)** under strict compute and memory constraints, designed for analyzing preference optimization behavior rather than large-scale training.

DPO avoids RL rollouts and reward model training, and instead directly optimizes a closed-form objective using preference pairs.

---

## Theory: Direct Preference Optimization

### Problem setup
For each prompt `x`, we observe two responses:
- `y+` = preferred (chosen)
- `y-` = less preferred (rejected)

We train a policy model `pi_theta(y | x)` to assign higher probability to `y+` than to `y-`, while staying close to a frozen reference model `pi_ref`.

### Key quantities (log-odds)
Define the policy preference margin:

Delta_pi = log pi_theta(y+ | x) - log pi_theta(y- | x)

Define the reference preference margin:

Delta_ref = log pi_ref(y+ | x) - log pi_ref(y- | x)

### DPO loss
DPO minimizes:

Loss = - log( sigmoid( beta * (Delta_pi - Delta_ref) ) )

where:
- `sigmoid(z) = 1 / (1 + exp(-z))`
- `beta` controls how strongly we deviate from the reference

### Intuition
- If the policy prefers `y+` over `y-` more than the reference does, then `(Delta_pi - Delta_ref)` is positive, sigmoid is near 1, and loss is small.
- If not, the loss increases and gradients push the policy to increase `pi_theta(y+ | x)` and/or decrease `pi_theta(y- | x)`, relative to the reference.

### Length normalization (important practical detail)
Summed token log-probs can bias toward shorter responses. We therefore compute response log-prob as an average over response tokens:

log pi(y | x) = (1 / T) * sum_t log pi(y_t | x, y_<t)

where `T` is the number of response tokens (excluding the prompt).

---

## Dataset

### Anthropic HH-RLHF
We use a subset of the **Anthropic Helpful–Harmless RLHF** dataset.

Each example contains:
- `chosen`: a full conversation transcript ending in a preferred assistant answer
- `rejected`: the same conversation context ending in a less preferred answer

There is no explicit `prompt` field.

### Prompt extraction
We extract `(prompt, response)` by splitting on the final `"\n\nAssistant:"` marker:
- `prompt` = everything up to and including the last `"Assistant:"`
- `response` = the assistant’s final answer after that marker

This ensures chosen and rejected are compared under identical context.

### Subsampling
To fit Colab constraints, we train on a small random subset (e.g., 200–300 preference pairs). This is sufficient for studying short-horizon optimization dynamics.

---

## Models

### Policy model
- Base model: `Qwen/Qwen2.5-0.5B`
- Training: LoRA adapters (rank 8) on attention projections (q_proj and v_proj)
- Precision: bfloat16
- Gradient checkpointing: enabled

Only LoRA parameters are updated during training.

### Reference model
- Same base model
- Loaded in 4-bit quantization (bitsandbytes)
- Frozen (no gradients)

The reference acts as a stable anchor and implicitly regularizes the policy.

---

## Implementation Notes

### Log-prob computation
- We build `input_ids = prompt + response`
- We mask prompt tokens in `labels` with `-100` so only response tokens contribute
- We compute token log-probs via cross-entropy and average over response tokens

### Optimization
- Batch size: 1 (memory-safe)
- Gradient accumulation: 8 (effective batch ~8)
- Optimizer: AdamW
- Gradient clipping: enabled
- Training horizon: a few hundred steps

---

## Scope
This repo is intended for:
- understanding DPO mechanics
- studying stability / preference margin behavior
- comparing against PPO / GRPO under controlled constraints

It is not a production-scale training setup.

---

## Future Extensions
- PPO and GRPO from scratch under the same preprocessing and logging
- KL drift + preference margin diagnostics
- Shortcut sensitivity tests (length / refusal / hedging bias)
- Out-of-domain preference evaluation

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
