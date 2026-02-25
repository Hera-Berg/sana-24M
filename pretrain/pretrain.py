"""
pretrain/pretrain.py — the model's pretraining loop

Features:
- BinShardDataset: mmap reader over pre-tokenized uint16 .bin shards (zero RAM copy)
- Cosine LR with linear warmup
- Mixed precision via torch.amp (bf16 preferred, fp16 with GradScaler otherwise)
- Gradient accumulation
- Grad clipping, weight decay on 2D+ params only
- Gradient checkpointing support
- torch.compile support
- Fused AdamW on CUDA
- --resume for interrupted runs
- Loss sanity warnings and periodic validation

Usage:
    python pretrain/pretrain.py --config pretrain/config.yaml
    python pretrain/pretrain.py --config pretrain/config.yaml --resume checkpoints/pretrain/ckpt_005000.pt
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

try:
    import yaml
except ImportError:
    raise ImportError("Please install pyyaml: pip install pyyaml")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Please install tqdm: pip install tqdm")


sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import SanaConfig, SanaModel
from tokenizer.tokenizer import Tokenizer


class BinShardDataset(IterableDataset):
    """
    Reads pre-tokenized uint16 .bin shards via np.memmap — no RAM copy,
    no tokenization at train time.  Each shard is a flat array of token IDs
    written by pretrain/tokenize_data.py.

    Chunks are (max_seq_len + 1)-token windows taken with a stride of
    max_seq_len, so adjacent windows share only the single boundary token
    (yielding non-overlapping input/target pairs).  Shards and the chunk
    order within each shard are shuffled every epoch via set_epoch().
    Workers shard across files so no chunk is produced twice.
    """

    def __init__(self, data_dir: str, max_seq_len: int, seed: int = 42):
        self.shards = sorted(Path(data_dir).glob("shard_*.bin"))
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.epoch = 0

        if not self.shards:
            raise FileNotFoundError(
                f"No shard_*.bin files found in {data_dir}. Run: python pretrain/tokenize_data.py first."
            )

        meta_path = Path(data_dir) / "meta.json"
        if meta_path.exists():
            import json as _json

            meta = _json.loads(meta_path.read_text())
            self.total_tokens = meta.get("total_tokens", 0)
            print(
                f"BinShardDataset: {len(self.shards)} shards, {self.total_tokens:,} total tokens"
            )
        else:

            self.total_tokens = sum(p.stat().st_size // 2 for p in self.shards)
            print(
                f"BinShardDataset: {len(self.shards)} shards, ~{self.total_tokens:,} tokens (estimated from file sizes)"
            )

    def val_shards(self, val_frac: float = 0.02):
        """Return the last val_frac of shards for validation (held out from training)."""
        n_val = max(1, int(len(self.shards) * val_frac))
        return self.shards[-n_val:]

    def train_shards(self, val_frac: float = 0.02):
        n_val = max(1, int(len(self.shards) * val_frac))
        return self.shards[:-n_val]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        rng = np.random.default_rng(self.seed + self.epoch)
        shards = list(self.train_shards())
        rng.shuffle(shards)

        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        T = self.max_seq_len

        for shard_path in shards:

            data = np.memmap(shard_path, dtype=np.uint16, mode="r")
            n_chunks = (len(data) - 1) // T

            idxs = np.arange(n_chunks, dtype=np.int64)
            rng.shuffle(idxs)

            for i in idxs:
                s = int(i) * T
                chunk = data[s : s + T + 1].astype(np.int64)
                x = torch.from_numpy(chunk[:-1].copy())
                y = torch.from_numpy(chunk[1:].copy())
                yield x, y

            del data


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0
) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    device: str,
    use_compile: bool,
) -> torch.optim.Optimizer:
    """
    Weight decay only on 2D+ parameters (weight matrices, not norms/biases).
    Use fused AdamW on CUDA when available.
    """
    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = (
        "cuda" in device
        and not use_compile
        and hasattr(torch.optim, "AdamW")
        and "fused" in torch.optim.AdamW.__init__.__doc__
        if torch.optim.AdamW.__init__.__doc__
        else False
    )
    use_fused = fused_available
    try:
        if "cuda" in device and not use_compile:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(beta1, beta2),
                fused=True,
            )
            print("Using fused AdamW")
            return optimizer
    except TypeError:
        pass

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(beta1, beta2))
    return optimizer


def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: SanaConfig,
    cfg_dict: dict,
    output_dir: str,
    tokenizer_path: str,
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"ckpt_{step:06d}.pt")
    torch.save(
        {
            "step": step,
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
    print(f"Checkpoint saved: {ckpt_path}")


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"]
    print(f"Resumed from step {step}: {path}")
    return step


def train(args):

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
        dropout=model_cfg.get("dropout", 0.1),
        rope_base=model_cfg.get("rope_base", 10000),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer_path = train_cfg["tokenizer_path"]
    tokenizer = Tokenizer(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    dataset = BinShardDataset(
        data_dir=train_cfg["data_dir"],
        max_seq_len=train_cfg["max_seq_len"],
        seed=train_cfg.get("seed", 42),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("dataloader_workers", 2),
        pin_memory="cuda" in device,
        drop_last=True,
        persistent_workers=train_cfg.get("dataloader_workers", 2) > 0,
    )

    batch_tokens = (
        train_cfg["batch_size"]
        * train_cfg.get("gradient_accumulation", 8)
        * train_cfg["max_seq_len"]
    )
    passes = train_cfg.get("passes", None)
    if passes is not None:
        steps_per_pass = dataset.total_tokens // batch_tokens
        max_steps = int(steps_per_pass * passes)
        print(
            f"Auto max_steps: {steps_per_pass:,} steps/pass × {passes} passes = {max_steps:,} steps"
        )
        print(
            f"Estimated time: ~{max_steps * batch_tokens / 20000 / 3600:.1f}h at 20k tok/s"
        )
    else:
        max_steps = train_cfg["max_steps"]
        print(f"max_steps from config: {max_steps:,}")

    model = SanaModel(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt:
        model.use_gradient_checkpointing = True
        print("Gradient checkpointing enabled (~30% VRAM saving, ~20% slower)")

    use_compile = train_cfg.get("use_compile", False) and not use_grad_ckpt
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = configure_optimizer(
        model=model,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        beta1=train_cfg["beta1"],
        beta2=train_cfg["beta2"],
        device=device,
        use_compile=use_compile,
    )

    use_bf16 = train_cfg.get("bf16", False) and "cuda" in device
    use_fp16 = train_cfg.get("fp16", True) and "cuda" in device and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = use_bf16 or use_fp16

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    if use_bf16:
        print("Precision: bf16 (no scaler needed)")
    elif use_fp16:
        print("Precision: fp16 (with GradScaler)")
    else:
        print("Precision: fp32")

    grad_accum = train_cfg.get("gradient_accumulation", 8)

    warmup_steps = train_cfg["warmup_steps"]
    grad_clip = train_cfg["grad_clip"]
    save_every = train_cfg["save_every"]
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("val_every", 2000)
    val_frac = train_cfg.get("val_frac", 0.02)
    output_dir = train_cfg["output_dir"]
    lr_max = train_cfg["learning_rate"]
    lr_min = lr_max * 0.1

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scaler)

    step = start_step
    epoch = 0
    model.train()
    dataset.set_epoch(epoch)
    data_iter = iter(dataloader)
    t_start = time.time()
    tokens_seen = 0

    print(f"\nStarting training from step {start_step}")
    print(
        f"Effective batch: {train_cfg['batch_size'] * grad_accum} seqs = "
        f"{train_cfg['batch_size'] * grad_accum * train_cfg['max_seq_len']:,} tokens"
    )
    print()

    while step < max_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                dataset.set_epoch(epoch)
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.split(":")[0], enabled=use_amp, dtype=amp_dtype
            ):
                _, loss = model(x, labels=y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item()

        tokens_seen += train_cfg["batch_size"] * grad_accum * train_cfg["max_seq_len"]

        lr = get_lr(step, warmup_steps, max_steps, lr_max, lr_min)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        step += 1
        step_time = time.time() - step_start
        elapsed = time.time() - t_start
        toks_per_s = tokens_seen / elapsed if elapsed > 0 else 0

        if step % log_every == 0 or step == 1:
            print(
                f"step {step:6d}/{max_steps} | "
                f"loss={total_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"grad={grad_norm:.3f} | "
                f"tok/s={toks_per_s:,.0f} | "
                f"elapsed={elapsed/60:.1f}m"
            )

        if step == 200 and total_loss > 7.5:
            warnings.warn(
                f"Loss {total_loss:.3f} > 7.5 at step 200. "
                "Possible pipeline bug (bad chunking, tokenizer issue, or untrained model)."
            )
        if step > 5000 and total_loss < 2.0:
            warnings.warn(
                f"Loss {total_loss:.3f} < 2.0 at step {step}. "
                "Possible memorisation or dataset too small."
            )

        if step % val_every == 0 or step == max_steps:
            model.eval()
            val_shards = dataset.val_shards(val_frac)
            val_loss_sum = 0.0
            val_batches = 0
            val_max_batches = 50
            with torch.no_grad():
                for shard_path in val_shards:
                    if val_batches >= val_max_batches:
                        break
                    data = np.memmap(shard_path, dtype=np.uint16, mode="r")
                    T = train_cfg["max_seq_len"]
                    n = (len(data) - 1) // T
                    for i in range(min(n, val_max_batches - val_batches)):
                        chunk = data[i * T : i * T + T + 1].astype(np.int64)
                        xv = torch.from_numpy(chunk[:-1].copy()).unsqueeze(0).to(device)
                        yv = torch.from_numpy(chunk[1:].copy()).unsqueeze(0).to(device)
                        with torch.amp.autocast(
                            device_type=device.split(":")[0],
                            enabled=use_amp,
                            dtype=amp_dtype,
                        ):
                            _, vloss = model(xv, labels=yv)
                        val_loss_sum += vloss.item()
                        val_batches += 1
                    del data
            val_loss = val_loss_sum / max(val_batches, 1)
            gap = val_loss - total_loss
            overfit_warn = " ⚠️  OVERFIT" if gap > 0.3 else ""
            print(
                f"  VAL step {step:6d} | val_loss={val_loss:.4f} | train_loss={total_loss:.4f} | gap={gap:+.4f}{overfit_warn}"
            )
            model.train()

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                cfg_dict=cfg_dict,
                output_dir=output_dir,
                tokenizer_path=tokenizer_path,
            )

    print(f"\nTraining complete! Steps: {step}, Tokens: {tokens_seen/1e9:.2f}B")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Sana 🪼")
    parser.add_argument("--config", type=str, default="pretrain/config.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

