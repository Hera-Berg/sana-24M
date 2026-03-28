"""
finetune/finetune.py — supervised fine-tuning (SFT) for the model

Loads a pretrained checkpoint and fine-tunes on a dialogue JSONL file
(path set via config). Loss is computed only on the model's own response
tokens; user tokens and structural markers are masked to -100 by the
tokenizer's build_multiturn_training_sequence().

Usage:
    python finetune/finetune.py --config finetune/config.yaml
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError:
    raise ImportError("Please install pyyaml: pip install pyyaml")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import SanaConfig, SanaModel
from tokenizer.tokenizer import Tokenizer


class DialogueDataset(Dataset):
    """
    Loads a dialogue JSONL file (one JSON object per line).
    Each line is expected to contain a "conversations": [...] list; only that
    field is read (any other fields, e.g. "_meta", are ignored). Lines that
    fail to parse or have no conversations are skipped.
    Uses tokenizer.build_multiturn_training_sequence() to produce the
    (input_ids, label_ids) pair with response-only loss masking.
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int):
        self.samples = []
        skipped = 0

        with open(data_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error on line {line_no}: {e}")
                    skipped += 1
                    continue

                conversations = obj.get("conversations", [])
                if not conversations:
                    skipped += 1
                    continue

                input_ids, label_ids = tokenizer.build_multiturn_training_sequence(
                    conversations=conversations,
                    max_seq_len=max_seq_len,
                )
                self.samples.append(
                    (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(label_ids, dtype=torch.long),
                    )
                )

        print(
            f"Loaded {len(self.samples)} dialogues ({skipped} skipped) from {data_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Stack pre-padded samples (all same length from build_multiturn_training_sequence)."""
    inputs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return inputs, labels


def get_lr(
    step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 0.0
) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def save_checkpoint(
    step_or_epoch,
    model,
    optimizer,
    scaler,
    config,
    output_dir,
    tokenizer_path,
    is_epoch=False,
):
    os.makedirs(output_dir, exist_ok=True)
    tag = f"epoch_{step_or_epoch:02d}" if is_epoch else f"step_{step_or_epoch:06d}"
    ckpt_path = os.path.join(output_dir, f"ckpt_{tag}.pt")
    torch.save(
        {
            "step": step_or_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        ckpt_path,
    )
    config.save(os.path.join(output_dir, "config.json"))

    tok_dest = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tok_dest):
        shutil.copy2(tokenizer_path, tok_dest)
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def finetune(args):

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["training"]

    config = SanaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_multiplier=model_cfg["ffn_multiplier"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg.get("dropout", 0.0),
        rope_base=model_cfg.get("rope_base", 10000),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer_path = train_cfg["tokenizer_path"]
    tokenizer = Tokenizer(tokenizer_path)
    print(f"Tokenizer loaded: vocab={len(tokenizer)}")

    dataset = DialogueDataset(
        data_path=train_cfg["data_path"],
        tokenizer=tokenizer,
        max_seq_len=train_cfg["max_seq_len"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = SanaModel(config).to(device)

    pretrain_ckpt = train_cfg.get("pretrain_checkpoint")
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"Loading pretrained weights from {pretrain_ckpt}...")
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")

        state_dict = ckpt.get("model", ckpt)

        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(
                f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )
        print("Pretrained weights loaded.")
    else:
        print(
            "WARNING: No pretrained checkpoint found. Training from scratch (not recommended)."
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt:
        model.use_gradient_checkpointing = True
        print("Gradient checkpointing enabled")

    freeze_layers_cfg = train_cfg.get("freeze_layers")
    freeze_layers = freeze_layers_cfg if freeze_layers_cfg is not None else 0
    for i, layer in enumerate(model.layers):
        if i < freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Frozen layers: {freeze_layers}/{len(model.layers)} | Trainable: {trainable/1e6:.2f}M | Frozen: {frozen_params/1e6:.2f}M"
    )

    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )

    use_fp16 = train_cfg.get("fp16", True) and "cuda" in device
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    num_epochs = train_cfg.get("num_epochs", 5)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    warmup_steps = train_cfg.get("warmup_steps", 50)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 10)
    save_every_ep = train_cfg.get("save_every_epoch", True)
    output_dir = train_cfg["output_dir"]
    lr_max = train_cfg["learning_rate"]
    lr_min = lr_max * 0.1

    total_steps = num_epochs * math.ceil(len(dataloader) / grad_accum)

    print(f"\nFine-tuning for {num_epochs} epochs, {len(dataset)} samples")
    print(f"Effective batch: {train_cfg['batch_size'] * grad_accum}")
    print(f"Total steps: {total_steps}\n")

    global_step = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.split(":")[0], enabled=use_fp16):
                _, loss = model(x, labels=y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):

                lr = get_lr(global_step, warmup_steps, total_steps, lr_max, lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_steps += 1

                current_loss = loss.item() * grad_accum
                if global_step % log_every == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"epoch {epoch}/{num_epochs} | "
                        f"step {global_step} | "
                        f"loss={current_loss:.4f} | "
                        f"lr={lr:.2e} | "
                        f"elapsed={elapsed/60:.1f}m"
                    )

                early_stop = train_cfg.get("early_stop_loss", None)
                if early_stop and current_loss < early_stop:
                    print(
                        f"\nEarly stop: loss {current_loss:.4f} < threshold {early_stop}"
                    )
                    save_checkpoint(
                        global_step,
                        model,
                        optimizer,
                        scaler,
                        config,
                        output_dir,
                        tokenizer_path,
                        is_epoch=False,
                    )
                    return

        avg_loss = (epoch_loss * grad_accum) / max(1, epoch_steps)
        print(f"\n>>> Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")

        if save_every_ep:
            save_checkpoint(
                step_or_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                output_dir=output_dir,
                tokenizer_path=tokenizer_path,
                is_epoch=True,
            )

    print(f"\nFine-tuning complete! 🪼")

    final_ckpt = save_checkpoint(
        step_or_epoch=num_epochs,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        is_epoch=True,
    )
    print(f"Final checkpoint: {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sana 🪼")
    parser.add_argument("--config", type=str, default="finetune/config.yaml")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
