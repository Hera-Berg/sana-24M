"""
test_pretrain.py — Evaluate Sana 🪼 pretrained model

Tests 25 prompts across 5 categories:
  1. Coherence baseline
  2. General science
  3. Biology
  4. Ocean & jellyfish
  5. Sentence continuation

Usage:
    python test_pretrain.py
    python test_pretrain.py --checkpoint checkpoints/pretrain --temperature 0.7
"""

import sys
import re
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.model import SanaModel, SanaConfig
from tokenizer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

TEST_PROMPTS = {
    "coherence": [
        "The quick brown fox",
        "Once upon a time, in a land far away,",
        "Scientists have long believed that",
        "The most important thing to remember is",
        "In conclusion, we can see that",
    ],
    "general_science": [
        "The speed of light in a vacuum is",
        "Gravity is the force that",
        "The periodic table organises elements by",
        "Stars generate energy through a process called",
        "The laws of thermodynamics describe",
    ],
    "biology": [
        "DNA carries genetic information in",
        "Cells are the basic unit of",
        "Evolution occurs through the mechanism of",
        "The mitochondria is responsible for",
        "Photosynthesis converts sunlight into",
    ],
    "ocean_jellyfish": [
        "The ocean covers approximately",
        "Marine ecosystems depend on",
        "Jellyfish belong to the phylum",
        "The deep ocean is characterised by",
        "Coral reefs are formed by",
    ],
    "continuation": [
        "Water boils at 100 degrees Celsius at",
        "The human brain contains approximately",
        "The Earth orbits the Sun once every",
        "Plants produce oxygen as a byproduct of",
        "The largest planet in our solar system is",
    ],
}


# ---------------------------------------------------------------------------
# Output scorer
# ---------------------------------------------------------------------------

# Factual keyword checks for known prompts
FACTUAL_CHECKS = {
    "Water boils at 100 degrees Celsius at":        ["sea", "normal", "standard", "atmospheric", "pressure"],
    "The Earth orbits the Sun once every":           ["year", "365", "month"],
    "The largest planet in our solar system is":    ["Jupiter", "jupiter"],
    "The human brain contains approximately":       ["billion", "neurons", "100", "86"],
    "Plants produce oxygen as a byproduct of":      ["photosynthesis", "light"],
    "Photosynthesis converts sunlight into":        ["energy", "glucose", "sugar", "food"],
    "DNA carries genetic information in":           ["cell", "nucleus", "sequence", "base", "chromosome"],
    "Jellyfish belong to the phylum":               ["Cnidaria", "cnidaria", "medus"],
    "The speed of light in a vacuum is":            ["300", "186", "299", "miles", "km", "meters"],
    "Stars generate energy through a process called": ["fusion", "nuclear"],
}


def score_output(text: str, prompt: str = "") -> str:
    """
    Heuristic pass/fail scorer for generated text.
    Returns "PASS" or "FAIL <reason>".
    """
    # Strip control characters and artifacts before scoring
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('\ufffd', '').replace('', '')

    if len(text) < 10:
        return "FAIL length < 10 chars"

    words = text.split()
    if len(words) < 5:
        return "FAIL word count < 5"

    if re.search(r'(.)1{4,}', text):
        return "FAIL repeated character (5+ consecutive)"

    allowed = set(' .,;:!?-\'"()\n')
    alphanumeric_and_allowed = sum(
        1 for ch in text if ch.isalnum() or ch in allowed
    )
    ratio_bad = 1.0 - (alphanumeric_and_allowed / max(1, len(text)))
    if ratio_bad > 0.15:
        return f"FAIL high non-alphanumeric ratio ({ratio_bad:.2%})"

    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2:
            return f"FAIL avg word length too short ({avg_word_len:.1f})"
        if avg_word_len > 15:
            return f"FAIL avg word length too long ({avg_word_len:.1f})"

    # Factual check: continuation must contain a correct keyword
    if prompt in FACTUAL_CHECKS:
        expected = FACTUAL_CHECKS[prompt]
        generated = text[len(prompt):]
        if not any(kw in generated for kw in expected):
            return f"FAIL factual: expected one of {expected[:3]}..."

    return "PASS"



# ---------------------------------------------------------------------------
# ASCII bar chart
# ---------------------------------------------------------------------------

def ascii_bar(label: str, passed: int, total: int, width: int = 20) -> str:
    ratio = passed / total if total > 0 else 0
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<20} [{bar}] {passed}/{total} ({ratio*100:.0f}%)"


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(args):
    print("=" * 65)
    print("Sana 🪼 Pretrain Evaluation")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config + tokenizer
    config_path = Path(args.checkpoint) / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {args.checkpoint}")
        sys.exit(1)

    config = SanaConfig.load(str(config_path))

    tok_path = Path(args.checkpoint) / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path("tokenizer/sana_tokenizer/tokenizer.json")
    tokenizer = Tokenizer(str(tok_path))

    # Load model
    model = SanaModel(config).to(device)
    model.eval()

    # Find checkpoint
    pt_files = sorted(Path(args.checkpoint).glob("*.pt"))
    if pt_files:
        ckpt = torch.load(str(pt_files[-1]), map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        # Strip _orig_mod. prefix added by torch.compile
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded: {pt_files[-1].name}")
    else:
        print("WARNING: No .pt checkpoint found — using random weights")

    print(f"Device: {device} | Temperature: {args.temperature} | Top-p: {args.top_p}\n")

    # Run tests
    category_results = {}
    all_results = []

    for category, prompts in TEST_PROMPTS.items():
        category_pass = 0
        print(f"\n[{category.upper().replace('_', ' ')}]")
        print("-" * 55)

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            with torch.inference_mode():
                output = model.generate(
                    input_ids          = input_tensor,
                    max_new_tokens     = 40,
                    temperature        = args.temperature,
                    top_p              = args.top_p,
                    eos_token_id       = Tokenizer.END_ID,
                    repetition_penalty = 1.1,
                )

            # Decode only new tokens
            new_ids = output[0, len(input_ids):].tolist()
            generated = tokenizer.decode(new_ids, skip_special=True)
            full_text  = prompt + generated

            result = score_output(full_text, prompt=prompt)
            passed = result == "PASS"
            if passed:
                category_pass += 1
            all_results.append(passed)

            # Display
            status = "✓" if passed else "✗"
            display_gen = generated[:60].replace('\n', ' ')
            print(f"  {status} Prompt: {prompt!r}")
            print(f"    → {display_gen!r}")
            if not passed:
                print(f"    ↳ {result}")

        category_results[category] = (category_pass, len(prompts))

    # Summary
    total_pass  = sum(all_results)
    total_tests = len(all_results)
    overall_pct = total_pass / total_tests * 100

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for cat, (passed, total) in category_results.items():
        print(ascii_bar(cat, passed, total))
    print("-" * 65)
    print(ascii_bar("OVERALL", total_pass, total_tests, width=30))

    print()
    threshold = 0.60
    if overall_pct >= threshold * 100:
        print(f"✓ PASSED ({overall_pct:.1f}% ≥ {threshold*100:.0f}% threshold)")
        print("  Model shows basic language coherence. Proceed to fine-tuning.")
    else:
        print(f"✗ BELOW THRESHOLD ({overall_pct:.1f}% < {threshold*100:.0f}%)")
        print("  Consider training longer or checking data pipeline.")


def main():
    parser = argparse.ArgumentParser(description="Test Sana 🪼 pretrained model")
    parser.add_argument("--checkpoint",   type=str,   default="checkpoints/pretrain")
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--top_p",        type=float, default=0.9)
    args = parser.parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
