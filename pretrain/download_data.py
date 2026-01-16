"""
pretrain/download_data.py — Download and filter FineWeb-Edu for the model's pretraining

Source: HuggingFaceFW/fineweb-edu, sample-10BT
Default target: 2B tokens (~1.4B words / ~1.8M articles)

Resume-safe: re-running skips already-complete files and continues from where
it left off.  A partial last file is overwritten to avoid double-counting.

Usage:
    python pretrain/download_data.py --max_tokens 2000 --output_dir data/pretrain_raw

Requirements:
    pip install datasets tqdm
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "Please install required packages: pip install datasets tqdm"
    ) from e


EDUCATIONAL_KEYWORDS = {
    "biology",
    "chemistry",
    "physics",
    "mathematics",
    "math",
    "science",
    "astronomy",
    "geology",
    "ecology",
    "genetics",
    "evolution",
    "anatomy",
    "cell",
    "molecule",
    "atom",
    "organism",
    "species",
    "experiment",
    "hypothesis",
    "theory",
    "research",
    "study",
    "analysis",
    "data",
    "history",
    "historical",
    "ancient",
    "civilization",
    "empire",
    "war",
    "revolution",
    "democracy",
    "government",
    "society",
    "culture",
    "religion",
    "philosophy",
    "economics",
    "politics",
    "geography",
    "population",
    "literature",
    "novel",
    "poem",
    "poetry",
    "language",
    "grammar",
    "writing",
    "author",
    "story",
    "narrative",
    "character",
    "theme",
    "symbol",
    "technology",
    "engineering",
    "computer",
    "software",
    "algorithm",
    "programming",
    "network",
    "electricity",
    "energy",
    "machine",
    "robot",
    "medicine",
    "medical",
    "health",
    "disease",
    "treatment",
    "patient",
    "symptoms",
    "diagnosis",
    "therapy",
    "surgery",
    "nutrition",
    "exercise",
    "education",
    "learning",
    "teaching",
    "school",
    "university",
    "student",
    "knowledge",
    "concept",
    "definition",
    "example",
    "principle",
    "law",
    "equation",
    "formula",
    "proof",
    "theorem",
    "lesson",
    "curriculum",
    "ocean",
    "marine",
    "coral",
    "reef",
    "fish",
    "whale",
    "shark",
    "jellyfish",
    "plankton",
    "habitat",
    "ecosystem",
    "climate",
    "environment",
    "explain",
    "describe",
    "defined",
    "refers",
    "involves",
    "consists",
    "process",
    "function",
    "structure",
    "system",
    "method",
    "technique",
}

SPAM_PATTERNS = [
    "buy now",
    "add to cart",
    "privacy policy",
    "cookie policy",
    "subscribe now",
    "free shipping",
    "limited time offer",
    "click here",
    "terms of service",
    "all rights reserved",
    "copyright ©",
    "sign up for",
    "newsletter",
    "unsubscribe",
    "spam",
    "advertisement",
    "sponsored",
    "affiliate",
    "discount code",
    "promo code",
]


def has_educational_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in EDUCATIONAL_KEYWORDS)


def is_spam_or_commercial(text: str) -> bool:
    snippet = text[:500].lower()
    return any(pat in snippet for pat in SPAM_PATTERNS)


def prose_ratio_ok(text: str) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    short_lines = sum(1 for l in lines if len(l.split()) < 5)
    return (short_lines / len(lines)) <= 0.35


def word_count_ok(text: str) -> bool:
    wc = len(text.split())
    return 120 <= wc <= 50_000


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.split()) >= 3 or l.strip() == ""]
    return "\n".join(lines).strip()


def passes_filters(record: dict) -> bool:
    text = record.get("text", "")
    if not text:
        return False
    if record.get("score", 0.0) < 2.0:
        return False
    if not word_count_ok(text):
        return False
    if not prose_ratio_ok(text):
        return False
    if not has_educational_keyword(text):
        return False
    if is_spam_or_commercial(text):
        return False
    return True


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)


def scan_existing(out_path: Path, articles_per_file: int):
    """
    Count already-complete files and their token total.

    A file is treated as "complete" if it contains at least articles_per_file
    articles (counted as non-empty segments split on double newlines). Any file
    with fewer articles is treated as partial: it is deleted so it can be
    rewritten from scratch, avoiding double-counting on resume. In normal use
    only the most recent file is ever partial.

    Returns:
        complete_files  – number of fully-written files
        skip_articles   – total articles already written in those files
        skip_tokens     – estimated tokens in those files
        resume_file_idx – next file index to write  (== complete_files)
    """
    files = sorted(out_path.glob("fineweb_*.txt"))
    if not files:
        return 0, 0, 0, 0

    complete_files = 0
    skip_articles = 0
    skip_tokens = 0

    for fpath in files:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        articles = text.split("\n\n")

        articles = [a.strip() for a in articles if a.strip()]
        n = len(articles)
        tok = sum(approx_tokens(a) for a in articles)

        if n >= articles_per_file:

            complete_files += 1
            skip_articles += n
            skip_tokens += tok
        else:

            print(
                f"  Partial file detected ({n} articles): {fpath.name} — will overwrite"
            )
            fpath.unlink()

    return complete_files, skip_articles, skip_tokens, complete_files


def download_data(
    output_dir: str,
    max_tokens_M: int = 2000,
    articles_per_file: int = 2000,
    seed: int = 42,
):
    max_tokens = max_tokens_M * 1_000_000
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {out_path} for existing files...")
    complete_files, skip_articles, skip_tokens, file_idx = scan_existing(
        out_path, articles_per_file
    )

    if complete_files:
        print(
            f"  Resuming: {complete_files} complete files already present "
            f"({skip_articles:,} articles, ~{skip_tokens/1e6:.1f}M tokens)"
        )
    else:
        print("  No existing files — starting fresh.")
    print()

    pct_done = skip_tokens / max_tokens * 100 if max_tokens > 0 else 100

    if pct_done >= 99.0:
        print(f"  {pct_done:.2f}% of target reached — close enough, treating as done.")
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"  Files:        {complete_files}")
        print(f"  Articles:     {skip_articles:,}")
        print(
            f"  Est. tokens:  {skip_tokens/1e6:.1f}M / {max_tokens_M}M ({pct_done:.2f}%)"
        )
        print(f"  Output dir:   {out_path}")
        return

    remaining_tokens = max_tokens - skip_tokens
    print(
        f"Target: {max_tokens_M}M tokens total  |  Still need: ~{remaining_tokens/1e6:.1f}M tokens"
    )
    print(f"Output: {out_path}")
    print()

    print("Loading FineWeb-Edu (streaming)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    articles_to_skip = skip_articles
    articles_skipped = 0
    in_skip_phase = articles_to_skip > 0

    article_buf = []
    total_tokens = skip_tokens
    total_articles = skip_articles
    total_scanned = 0
    t_start = time.time()
    done = False

    try:
        for record in dataset:
            total_scanned += 1

            if not passes_filters(record):
                continue

            if in_skip_phase:
                articles_skipped += 1
                if articles_skipped >= articles_to_skip:
                    in_skip_phase = False
                    print(
                        f"  Fast-forwarded past {articles_skipped:,} existing articles."
                    )
                continue

            text = clean_text(record["text"])
            if not text:
                continue

            n_tok = approx_tokens(text)
            article_buf.append(text)
            total_tokens += n_tok
            total_articles += 1

            if total_scanned % 500_000 == 0:
                elapsed = time.time() - t_start
                rate = total_articles / elapsed if elapsed > 0 else 0
                print(
                    f"  Scanned: {total_scanned:>9,} | "
                    f"Kept: {total_articles:>9,} | "
                    f"Tokens: {total_tokens/1e6:>7.1f}M | "
                    f"Rate: {rate:.1f} art/s | "
                    f"Elapsed: {elapsed/60:.1f}m"
                )

            if len(article_buf) >= articles_per_file:
                fname = out_path / f"fineweb_{file_idx:05d}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(article_buf))
                file_idx += 1
                article_buf = []

                if file_idx % 10 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"  Files written: {file_idx} | "
                        f"Articles: {total_articles:,} | "
                        f"Tokens: {total_tokens/1e6:.1f}M / {max_tokens_M}M"
                    )

            if total_tokens >= max_tokens:
                print(f"\nTarget reached: {total_tokens/1e6:.1f}M tokens")
                done = True
                break

        if article_buf:
            fname = out_path / f"fineweb_{file_idx:05d}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write("\n\n".join(article_buf))
            file_idx += 1

    finally:

        try:
            del dataset
        except Exception:
            pass
        gc.collect()

    elapsed = time.time() - t_start
    status = "complete" if done else "exhausted dataset"
    print(f"\n{'='*60}")
    print(f"Download {status}!")
    print(f"  Files:         {file_idx}")
    print(f"  Articles:      {total_articles:,}")
    print(f"  Est. tokens:   {total_tokens/1e6:.1f}M")
    print(f"  Scanned:       {total_scanned:,}")
    if total_scanned > 0:
        kept = total_articles - skip_articles
        scanned_new = total_scanned
        if scanned_new > 0:
            print(f"  Acceptance:    {kept/scanned_new*100:.1f}% (this run)")
    print(f"  Time:          {elapsed/60:.1f} minutes")
    print(f"  Output dir:    {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter FineWeb-Edu for Sana pretraining"
    )
    parser.add_argument("--output_dir", type=str, default="data/pretrain_raw")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Target token count in millions (default: 2000 = 2B)",
    )
    parser.add_argument("--articles_per_file", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_data(
        output_dir=args.output_dir,
        max_tokens_M=args.max_tokens,
        articles_per_file=args.articles_per_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
    os._exit(0)
