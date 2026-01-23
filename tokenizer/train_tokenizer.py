"""
tokenizer/train_tokenizer.py — Pure Python BPE tokenizer training for the model

Implements BPE from scratch using only the Python standard library.
No tokenizers, no sentencepiece.

Performance note: Training runs on a 100k-line sample of the corpus,
which is sufficient for high-quality 16k-vocab BPE merges while remaining
tractable. Full corpus would require days; sampled training converges in
~30 minutes for 16k merges.

Usage:
    python tokenizer/train_tokenizer.py --data_dir data/pretrain_raw \
        --vocab_size 16000 --sample_lines 100000
"""

import argparse
import heapq
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Set, Tuple

SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|user|>": 1,
    "<|sana|>": 2,
    "<|end|>": 3,
    "<tool>": 4,
    "</tool>": 5,
    "<tool_result>": 6,
    "</tool_result>": 7,
    "<sana_salute>": 8,
    "<sana_happy>": 9,
    "<sana_think>": 10,
    "<sana_sad>": 11,
    "<sana_smug>": 12,
    "Sana": 13,
}

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)


def pretokenise(text: str) -> List[List[str]]:
    """
    Split text into words. Each word becomes a list of characters/bytes.
    Non-ASCII bytes are represented as 2-hex-char strings (e.g. 'c3').
    Whitespace is preserved as space characters.
    """

    for special in SPECIAL_TOKENS:
        text = text.replace(special, " ")
    words = re.findall(r"\w+|[^\w\s]|\s+", text)
    result = []
    for word in words:
        chars: List[str] = []
        for ch in word:
            if ord(ch) < 128:
                chars.append(ch)
            else:
                for byte in ch.encode("utf-8"):
                    chars.append(f"{byte:02x}")
        if chars:
            result.append(chars)
    return result


def word_to_key(chars: List[str]) -> Tuple[str, ...]:
    return tuple(chars)


def get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    """Count all adjacent pairs across all words, weighted by word frequency."""
    counts: Counter = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def merge_vocab(
    word_freqs: Dict[Tuple[str, ...], int],
    pair: Tuple[str, str],
) -> Dict[Tuple[str, ...], int]:
    """
    Apply one BPE merge to the word frequency table.
    Replaces every occurrence of `pair` with the merged token.
    Returns (new_word_freqs, old_to_new) where old_to_new maps every
    changed old tuple → its new tuple (needed for efficient count updates).
    """
    a, b = pair
    merged = a + b
    new_vocab: Dict[Tuple[str, ...], int] = {}
    old_to_new: Dict[Tuple[str, ...], Tuple[str, ...]] = {}

    for word, freq in word_freqs.items():
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_tuple = tuple(new_word)
        new_vocab[new_tuple] = freq
        if new_tuple != word:
            old_to_new[word] = new_tuple

    return new_vocab, old_to_new


def update_pair_counts(
    pair_counts: Counter,
    word_freqs: Dict[Tuple[str, ...], int],
    old_to_new: Dict[Tuple[str, ...], Tuple[str, ...]],
) -> Counter:
    """
    Efficiently update pair counts after a merge.
    old_to_new maps every changed old-word tuple → its new merged tuple.
    """
    for old_word, new_word in old_to_new.items():
        freq = word_freqs[old_word]

        for i in range(len(old_word) - 1):
            pair_counts[(old_word[i], old_word[i + 1])] -= freq

        for i in range(len(new_word) - 1):
            pair_counts[(new_word[i], new_word[i + 1])] += freq

    return pair_counts


def _pretokenise_chunk(lines: List[str]) -> Counter:
    """Worker: pretokenise a chunk of lines, return word freq Counter."""
    counts: Counter = Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for word_chars in pretokenise(line):
            counts[word_to_key(word_chars)] += 1
    return counts


def train_bpe(
    corpus_lines: List[str],
    target_vocab_size: int,
    verbose: bool = True,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE on a list of text lines.
    Returns (vocab dict, list of merge pairs).
    Vocab IDs start at NUM_SPECIAL_TOKENS (14).
    """
    n_workers = max(1, cpu_count())
    print(f"Pre-tokenising {len(corpus_lines):,} lines with {n_workers} workers...")
    t0 = time.time()

    chunk_size = max(1, len(corpus_lines) // n_workers)
    chunks = [
        corpus_lines[i : i + chunk_size]
        for i in range(0, len(corpus_lines), chunk_size)
    ]

    word_freq_raw: Counter = Counter()
    with Pool(n_workers) as pool:
        for partial in pool.map(_pretokenise_chunk, chunks):
            word_freq_raw.update(partial)

    print(
        f"Pre-tokenisation done in {time.time()-t0:.1f}s. Unique words: {len(word_freq_raw):,}"
    )

    char_vocab: Set[str] = set()
    for word in word_freq_raw:
        for ch in word:
            char_vocab.add(ch)

    vocab: Dict[str, int] = {}
    next_id = NUM_SPECIAL_TOKENS

    for ch in sorted(char_vocab):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1

    num_initial = len(vocab) + NUM_SPECIAL_TOKENS
    num_merges_needed = target_vocab_size - num_initial
    print(
        f"Initial vocab size: {num_initial} ({len(char_vocab)} chars + {NUM_SPECIAL_TOKENS} special)"
    )
    print(f"Merges needed: {num_merges_needed}")

    if num_merges_needed <= 0:
        print(
            "Warning: target vocab_size already satisfied by character vocabulary alone."
        )
        return vocab, []

    word_freqs: Dict[Tuple[str, ...], int] = dict(word_freq_raw)
    pair_counts = get_pair_counts(word_freqs)

    heap = [(-cnt, p) for p, cnt in pair_counts.items()]
    heapq.heapify(heap)

    merges: List[Tuple[str, str]] = []
    t_start = time.time()

    for step in range(num_merges_needed):

        best_pair = None
        best_freq = 0
        while heap:
            neg_cnt, pair = heapq.heappop(heap)
            current_cnt = pair_counts.get(pair, 0)
            if current_cnt >= 2 and current_cnt == -neg_cnt:
                best_pair = pair
                best_freq = current_cnt
                break

        if best_pair is None:
            print(f"Stopping early at step {step}: no pair appears ≥2 times")
            break

        a, b = best_pair
        merged = a + b
        merges.append(best_pair)
        vocab[merged] = next_id
        next_id += 1

        new_word_freqs, old_to_new = merge_vocab(word_freqs, best_pair)
        pair_counts = update_pair_counts(pair_counts, word_freqs, old_to_new)
        word_freqs = new_word_freqs

        for old_word, new_word in old_to_new.items():
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                cnt = pair_counts.get(p, 0)
                if cnt > 0:
                    heapq.heappush(heap, (-cnt, p))

        if step % 2000 == 0:
            pair_counts = Counter({k: v for k, v in pair_counts.items() if v > 0})
            heap = [(-cnt, p) for p, cnt in pair_counts.items()]
            heapq.heapify(heap)

        if verbose and (step + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed
            eta = (num_merges_needed - step - 1) / rate if rate > 0 else 0
            print(
                f"  [{step+1:6d}/{num_merges_needed}] "
                f"merge: {a!r}+{b!r}→{merged!r} "
                f"(freq={best_freq}) "
                f"| {rate:.1f} merges/s | ETA {eta/60:.1f}m"
            )

    print(
        f"BPE training complete. {len(merges)} merges, vocab size = {len(vocab) + NUM_SPECIAL_TOKENS}"
    )
    return vocab, merges


def sample_corpus_lines(
    data_dir: str,
    max_lines: int = 100_000,
    seed: int = 42,
) -> List[str]:
    """
    Randomly sample up to max_lines lines from all .txt files in data_dir.
    Uses reservoir sampling so we don't need to read everything into memory.
    """
    rng = random.Random(seed)
    reservoir: List[str] = []
    n_seen = 0

    data_path = Path(data_dir)
    files = sorted(data_path.glob("**/*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    print(f"Found {len(files)} .txt files. Sampling up to {max_lines:,} lines...")

    for fp in files:
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if len(line.strip()) < 10:
                        continue
                    n_seen += 1
                    if len(reservoir) < max_lines:
                        reservoir.append(line)
                    else:
                        j = rng.randint(0, n_seen - 1)
                        if j < max_lines:
                            reservoir[j] = line
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")

    rng.shuffle(reservoir)
    print(f"Sampled {len(reservoir):,} lines from {n_seen:,} total lines")
    return reservoir


def sample_dialogue_lines(
    dialogue_files: list,
    max_lines: int = 20_000,
    seed: int = 42,
) -> List[str]:
    """
    Extract text lines from SFT .jsonl dialogue files so conversational
    words like the model name, 'hi', 'yes', 'compact' get enough frequency
    to merge into single BPE tokens.
    """
    import json as _json

    rng = random.Random(seed)
    lines = []
    for fpath in dialogue_files:
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                        for turn in obj.get("conversations", []):
                            content = turn.get("content", "").strip()
                            if len(content) < 5:
                                continue

                            for tok in [
                                "<sana_salute>",
                                "<sana_happy>",
                                "<sana_think>",
                                "<sana_sad>",
                                "<sana_smug>",
                                "<tool>",
                                "</tool>",
                                "<tool_result>",
                                "</tool_result>",
                            ]:
                                content = content.replace(tok, "")
                            content = content.strip()
                            if len(content) >= 5:
                                lines.append(content)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: could not read {fpath}: {e}")
    rng.shuffle(lines)
    result = lines[:max_lines]
    print(f"Sampled {len(result):,} dialogue lines from {len(dialogue_files)} file(s)")
    return result


def resolve_output_path(output: str) -> str:
    """Ensure output path points to a .json file, not a directory."""
    p = Path(output)
    if p.suffix != ".json":
        p = p / "tokenizer.json"
    return str(p)


def is_complete_tokenizer(path: str) -> bool:
    """Return True if a valid, complete tokenizer JSON already exists at path."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return (
            "vocab" in data
            and "merges" in data
            and len(data["vocab"]) > 100
            and len(data["merges"]) > 100
        )
    except Exception:
        return False


def save_tokenizer(
    vocab: Dict[str, int],
    merges: List[Tuple[str, str]],
    target_vocab_size: int,
    output_path: str,
):
    """Save tokenizer in the canonical JSON format."""
    full_vocab: Dict[str, int] = {}
    for token, id_ in SPECIAL_TOKENS.items():
        full_vocab[token] = id_
    full_vocab.update(vocab)

    data = {
        "version": "1.0",
        "vocab_size": target_vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "vocab": full_vocab,
        "merges": [[a, b] for a, b in merges],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Tokenizer saved to {output_path}")
    print(f"  Vocab entries: {len(full_vocab):,}")
    print(f"  Merges:        {len(merges):,}")


def run_sanity_check(tokenizer_path: str):
    """Run post-training sanity check on the saved tokenizer."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer import Tokenizer

    tok = Tokenizer(tokenizer_path)

    ids = tok.encode("hello world")
    assert Tokenizer.END_ID not in ids, (
        f"END token (ID={Tokenizer.END_ID}) must not appear in plain encode(). "
        f"Got ids: {ids}"
    )

    test_str = "Hello, world!"
    decoded = tok.decode(tok.encode(test_str))
    assert (
        decoded == test_str
    ), f"Round-trip decode failed: encode→decode of {test_str!r} gave {decoded!r}"

    special_ids = tok.encode("<sana_think>")
    assert special_ids == [
        Tokenizer.THINK_ID
    ], f"Special token encoding wrong: expected [{Tokenizer.THINK_ID}], got {special_ids}"

    mixed = tok.encode("Hello <sana_happy> world")
    assert Tokenizer.HAPPY_ID in mixed, "HAPPY_ID should be in encoded mixed string"
    assert Tokenizer.END_ID not in mixed, "END_ID must not appear in mixed encode"

    hello_ids = tok.encode("hello world")
    assert len(hello_ids) <= 4, (
        f"Vocab too small or merges missing: 'hello world' gave {len(hello_ids)} tokens {hello_ids}. "
        f"Expected ≤4 for a 16k BPE (got character-level tokenizer?)."
    )

    import json as _json

    data = _json.loads(open(tokenizer_path).read())
    actual_merges = len(data.get("merges", []))
    actual_vocab = len(data.get("vocab", {}))
    assert actual_merges > 1000, (
        f"Only {actual_merges} merges found — expected ~15790 for 16k vocab. ",
        f"Stub tokenizer? Delete tokenizer.json and retrain.",
    )
    assert actual_vocab > 1000, f"Only {actual_vocab} vocab entries — expected ~16000."

    print("Tokenizer sanity check passed ✓")
    print(f"  Vocab entries:  {actual_vocab:,}")
    print(f"  Merges:         {actual_merges:,}")
    print(f"  encode('hello world') = {hello_ids}  ({len(hello_ids)} tokens)")
    print(
        f"  encode('Hello, world!') round-trips: {tok.decode(tok.encode('Hello, world!'))!r}"
    )
    print(
        f"  encode('photosynthesis') = {tok.encode('photosynthesis')}  ({len(tok.encode('photosynthesis'))} tokens)"
    )
    sana_ids = tok.encode("Sana")
    print(
        f"  encode('Sana') = {sana_ids}  ({len(sana_ids)} token{'s' if len(sana_ids)>1 else ''})",
        end=" ",
    )
    if len(sana_ids) == 1 and sana_ids[0] == 13:
        print("✓ correct — single token ID 13")
    else:
        print(f"⚠ expected [13], got {sana_ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Sana BPE tokenizer from scratch"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/pretrain_raw",
        help="Directory containing pretokenised .txt files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16000,
        help="Target vocabulary size (including special tokens)",
    )
    parser.add_argument(
        "--sample_lines",
        type=int,
        default=100_000,
        help="Number of lines to sample for BPE training (100k is sufficient)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer/sana_tokenizer/tokenizer.json",
        help="Output path for tokenizer JSON (directory or .json file)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-train even if a complete tokenizer already exists",
    )
    parser.add_argument(
        "--dialogue_dir",
        type=str,
        default=None,
        help="Directory with SFT .jsonl files to mix in (ensures conversational words merge)",
    )
    parser.add_argument(
        "--dialogue_lines",
        type=int,
        default=20_000,
        help="Max dialogue lines to mix in (default: 20000)",
    )
    args = parser.parse_args()

    output_path = resolve_output_path(args.output)

    print("=" * 60)
    print("Sana 🪼 BPE Tokenizer Training")
    print("=" * 60)
    print(f"Target vocab size : {args.vocab_size}")
    print(
        f"Sample lines      : {args.sample_lines:,} educational + {args.dialogue_lines:,} dialogue"
    )
    print(f"Data directory    : {args.data_dir}")
    print(f"Dialogue dir      : {args.dialogue_dir or 'none'}")
    print(f"Output            : {output_path}")
    print()

    if not args.force and is_complete_tokenizer(output_path):
        print(f"Complete tokenizer already found at {output_path}")
        print("Skipping training. Pass --force to retrain.")
        print("\nRunning sanity check...")
        run_sanity_check(output_path)
        print("\nTokenizer training complete! 🪼")
        return

    edu_lines = sample_corpus_lines(
        data_dir=args.data_dir,
        max_lines=args.sample_lines,
        seed=args.seed,
    )

    dialogue_lines = []
    if args.dialogue_dir:
        jsonl_files = sorted(Path(args.dialogue_dir).glob("**/*.jsonl"))
        if jsonl_files:
            dialogue_lines = sample_dialogue_lines(
                dialogue_files=[str(f) for f in jsonl_files],
                max_lines=args.dialogue_lines,
                seed=args.seed,
            )
        else:
            print(f"Warning: no .jsonl files found in {args.dialogue_dir}")

    lines = edu_lines + dialogue_lines
    random.Random(args.seed).shuffle(lines)
    if dialogue_lines:
        print(
            f"Total corpus: {len(lines):,} lines ({len(edu_lines):,} edu + {len(dialogue_lines):,} dialogue)"
        )

    vocab, merges = train_bpe(
        corpus_lines=lines,
        target_vocab_size=args.vocab_size,
        verbose=True,
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        target_vocab_size=args.vocab_size,
        output_path=output_path,
    )

    print("\nRunning sanity check...")
    run_sanity_check(output_path)

    print("\nTokenizer training complete! 🪼")


if __name__ == "__main__":
    main()
