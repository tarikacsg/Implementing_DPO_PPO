# Implementing_DPO_PPO
DPO (Direct Preference Optimization) Implements Direct Preference Optimization, aligning language models using pairwise preference data without reward models or PPO.  PPO (Proximal Policy Optimization) Implements Proximal Policy Optimization for RLHF, fine-tuning language models with a reward model and KL-constrained updates.

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

## Purpose

This project focuses on understanding RLHF by implementing each component explicitly, without relying on high-level alignment frameworks. The emphasis is on clarity, correctness, and learning how modern LLM alignment works end to end.
