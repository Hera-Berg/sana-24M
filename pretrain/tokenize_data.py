"""
pretrain/tokenize_data.py — Pre-tokenize FineWeb-Edu .txt files into binary shards

Reads every .txt in --data_dir, tokenizes each file's full text in one pass,
appends a single END_ID per file, and writes the token stream to uint16 .bin
shards under --output_dir.  Each shard is a flat numpy uint16 array (~200MB
each by default) that pretrain.py mmaps directly — zero tokenization at train
time.

A small metadata JSON is written alongside each shard, plus a master meta.json,
so pretrain.py knows total token counts without scanning the files.

Usage:
    python pretrain/tokenize_data.py \
        --data_dir  data/pretrain_raw \
        --output_dir data/pretrain_bin \
        --shard_size 200          # MB per shard (default 200)
        --workers   4             # parallel workers (default: cpu_count)

Output layout:
    data/pretrain_bin/
        shard_00000.bin   # flat uint16 token array
        shard_00000.json  # {"tokens": N, "source_files": [...]}
        shard_00001.bin
        shard_00001.json
        ...
        meta.json         # total tokens, total shards, vocab_size
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.tokenizer import Tokenizer


def _clean_for_tokenize(text: str) -> str:
    """Strip non-ASCII that wasn't cleaned at download time (e.g. older data)."""
    UNICODE_MAP = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2014": " -- ",
        "\u2013": " - ",
        "\u2012": " - ",
        "\u2026": "...",
        "\u00a0": " ",
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2002": " ",
        "\u2003": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2212": "-",
        "\u2022": "-",
        "\u2192": "->",
    }
    for uni, asc in UNICODE_MAP.items():
        text = text.replace(uni, asc)

    import re as _re

    text = _re.sub(r"[^\x09\x0a\x20-\x7e]", "", text)
    return text


def _tokenize_file(args):
    """Called in a worker process. Returns (filepath_str, uint16_array)."""
    fpath_str, tok_path = args
    fpath = Path(fpath_str)
    try:
        tok = Tokenizer(tok_path)
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        text = _clean_for_tokenize(text)
        ids = tok.encode(text)
        if not ids:
            return fpath_str, None
        ids.append(tok.END_ID)
        return fpath_str, np.array(ids, dtype=np.uint16)
    except Exception as e:
        print(f"  WARN: failed to tokenize {fpath.name}: {e}", flush=True)
        return fpath_str, None


def tokenize_data(
    data_dir: str,
    output_dir: str,
    tok_path: str,
    shard_mb: int = 200,
    n_workers: int = 0,
):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_path.glob("**/*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    tokens_per_shard = (shard_mb * 1024 * 1024) // 2

    n_workers = n_workers or max(1, cpu_count() - 1)
    print(f"Tokenizing {len(files)} files → {out_path}")
    print(f"Shard size: {shard_mb} MB ({tokens_per_shard:,} tokens/shard)")
    print(f"Workers:    {n_workers}")
    print()

    t_start = time.time()
    shard_idx = 0
    shard_buf = []
    shard_tokens = 0
    shard_files = []
    total_tokens = 0
    total_files = 0
    skipped = 0

    def flush_shard():
        nonlocal shard_idx, shard_buf, shard_tokens, shard_files
        if not shard_buf:
            return
        arr = np.concatenate(shard_buf).astype(np.uint16)
        bpath = out_path / f"shard_{shard_idx:05d}.bin"
        arr.tofile(bpath)
        jpath = out_path / f"shard_{shard_idx:05d}.json"
        jpath.write_text(
            json.dumps(
                {
                    "tokens": int(len(arr)),
                    "source_files": [Path(f).name for f in shard_files],
                },
                indent=2,
            )
        )
        print(f"  Shard {shard_idx:05d}: {len(arr):>12,} tokens  →  {bpath.name}")
        shard_idx += 1
        shard_buf = []
        shard_tokens = 0
        shard_files = []

    work = [(str(f), tok_path) for f in files]

    with Pool(n_workers) as pool:
        for fpath_str, arr in pool.imap(_tokenize_file, work, chunksize=4):
            if arr is None:
                skipped += 1
                continue

            shard_buf.append(arr)
            shard_tokens += len(arr)
            total_tokens += len(arr)
            total_files += 1
            shard_files.append(fpath_str)

            if shard_tokens >= tokens_per_shard:
                flush_shard()

            if total_files % 100 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  Files: {total_files:>5} / {len(files)} | "
                    f"Tokens: {total_tokens/1e6:>8.1f}M | "
                    f"Elapsed: {elapsed/60:.1f}m",
                    flush=True,
                )

    flush_shard()

    meta = {
        "total_tokens": int(total_tokens),
        "total_shards": shard_idx,
        "total_files": total_files,
        "skipped_files": skipped,
        "vocab_size": 16000,
        "dtype": "uint16",
        "end_id": 3,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Tokenization complete!")
    print(f"  Source files:  {total_files:,}  ({skipped} skipped)")
    print(f"  Total tokens:  {total_tokens/1e9:.3f}B")
    print(f"  Shards:        {shard_idx}")
    print(f"  Output dir:    {out_path}")
    print(f"  Time:          {elapsed/60:.1f} minutes")
    print(f"  Speed:         {total_tokens/elapsed/1e6:.2f}M tokens/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize data for Sana pretraining"
    )
    parser.add_argument("--data_dir", type=str, default="data/pretrain_raw")
    parser.add_argument("--output_dir", type=str, default="data/pretrain_bin")
    parser.add_argument(
        "--tok_path", type=str, default="tokenizer/sana_tokenizer/tokenizer.json"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=200,
        help="Target shard size in MB (default: 200)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (default: cpu_count - 1)",
    )
    args = parser.parse_args()

    tokenize_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tok_path=args.tok_path,
        shard_mb=args.shard_size,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
