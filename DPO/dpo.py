import argparse
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model


# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
):
    # DPO: -log σ(β[(πc-πr) - (refc-refr)])
    chosen_rel = pi_chosen - ref_chosen
    rejected_rel = pi_rejected - ref_rejected
    logits = beta * (chosen_rel - rejected_rel)
    loss = -F.logsigmoid(logits).mean()

    acc = (chosen_rel > rejected_rel).float().mean()
    margin = (chosen_rel - rejected_rel).mean()
    return loss, acc, margin


def sequence_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
):
    """
    Average log-prob over RESPONSE tokens only (after prompt).
    Correctly shifts logits/labels for causal LM.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B,T,V]

    # Shift for causal LM
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    attn = attention_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    B, Tm1 = labels.shape
    positions = torch.arange(Tm1, device=input_ids.device).unsqueeze(0).expand(B, -1)

    # After shift, boundary is prompt_len - 1
    boundary = (prompt_lens - 1).clamp(min=0)
    response_mask = (positions >= boundary.unsqueeze(1)).float() * attn.float()

    denom = response_mask.sum(dim=1).clamp(min=1.0)
    return (token_logp * response_mask).sum(dim=1) / denom


# ----------------------------
# Data
# ----------------------------
@dataclass
class DPOBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    prompt_lens: torch.Tensor


class DPOCollator:
    """
    Builds prompt+chosen and prompt+rejected, pads them, and returns prompt lengths
    (so we can score only the response).
    """

    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        chosens = []
        rejecteds = []

        # Support both schemas:
        # 1) prompt/chosen/rejected
        # 2) instruction/input/output (where output is [chosen, rejected] or dict-like)
        for ex in batch:
            if "chosen" in ex and "rejected" in ex and "prompt" in ex:
                prompt = ex["prompt"]
                chosen = ex["chosen"]
                rejected = ex["rejected"]
            else:
                # common alpaca-style fallback
                instruction = ex.get("instruction", "")
                inp = ex.get("input", "")
                output = ex.get("output", None)
                if inp:
                    instruction = instruction + "\n" + inp

                if isinstance(output, (list, tuple)) and len(output) >= 2:
                    chosen, rejected = output[0], output[1]
                else:
                    raise ValueError("Example missing (prompt, chosen, rejected) OR (instruction, input, output[2]).")

                prompt = instruction

            # Simple template (edit as you like)
            p = f"Instruct: {prompt.strip()}\nOutput:"
            prompts.append(p)
            chosens.append(chosen.strip())
            rejecteds.append(rejected.strip())

        # Prompt lengths without padding (true boundary)
        prompt_tok = self.tok(prompts, add_special_tokens=False)
        prompt_lens = torch.tensor([len(ids) for ids in prompt_tok["input_ids"]], dtype=torch.long)

        chosen_texts = [p + " " + c for p, c in zip(prompts, chosens)]
        rejected_texts = [p + " " + r for p, r in zip(prompts, rejecteds)]

        chosen_enc = self.tok(
            chosen_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        rejected_enc = self.tok(
            rejected_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        # If prompt itself got truncated, clamp prompt_lens to sequence length
        seq_len = chosen_enc["input_ids"].shape[1]
        prompt_lens = torch.clamp(prompt_lens, max=seq_len)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "prompt_lens": prompt_lens,
        }


# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="microsoft/phi-2")
    ap.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--output_dir", type=str, default="model-DPO")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2003)
    ap.add_argument("--grad_accum", type=int, default=1)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Quantization
    ap.add_argument("--load_in_4bit", action="store_true")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Policy model (trainable)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.load_in_4bit else None,
        quantization_config=quant_cfg,
    )

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.load_in_4bit else None,
        quantization_config=quant_cfg,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # LoRA on policy model
    target_modules = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if not args.load_in_4bit:
        model.to(device)
        ref_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = load_dataset(args.dataset_name, split=args.split)
    collator = DPOCollator(tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            pi_chosen = sequence_logprob(
                model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["prompt_lens"]
            )
            pi_rejected = sequence_logprob(
                model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["prompt_lens"]
            )

            with torch.no_grad():
                ref_chosen = sequence_logprob(
                    ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["prompt_lens"]
                )
                ref_rejected = sequence_logprob(
                    ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["prompt_lens"]
                )

            loss, acc, margin = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=args.beta)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 25 == 0:
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"reward_acc={acc.item():.3f} margin={margin.item():.4f}"
                )
            step += 1

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
