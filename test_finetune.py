"""
test_finetune.py — Evaluate Sana 🪼 fine-tuned model

Tests:
  - 10 single and multi-turn dialogue tests
  - Per response checks: non-empty, no leaked <|user|>, ≤200 tokens, starts with emotion token
  - Bioluminescence experiment: model describes it but word absent from training data
  - Tool use test: model uses <tool>...</tool> for lookup requests

Usage:
    python test_finetune.py
    python test_finetune.py --checkpoint checkpoints/finetune
"""

import sys
import re
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.model import SanaModel, SanaConfig
from tokenizer.tokenizer import Tokenizer
from inference.inference import SanaInference


# ---------------------------------------------------------------------------
# Emotion token set (as they appear in decoded text)
# ---------------------------------------------------------------------------

EMOTION_TOKENS = {
    "<sana_salute>", "<sana_happy>", "<sana_think>", "<sana_sad>", "<sana_smug>"
}

EMOTION_EMOJIS = {"🪼", "✨", "🤔", "😔", "😏"}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_response(response: str, max_new_tokens: int = 200) -> dict:
    """
    Run all checks on a response.
    Returns dict: {check_name: bool}
    """
    results = {}

    # (a) Non-empty
    results["non_empty"] = len(response.strip()) > 0

    # (b) No <|user|> leaked into output
    results["no_user_leak"] = "<|user|>" not in response and "USER_ID" not in response

    # (c) Token count ≤ max_new_tokens (approximate via word count * 1.5)
    approx_tokens = len(response.split()) * 1.5
    results["within_token_limit"] = approx_tokens <= max_new_tokens

    # (d) Starts with emotion token (either raw form or emoji form after cleaning)
    # Check the beginning of the response (first 20 chars)
    stripped = response.strip()
    starts_emotion = False
    for tok in EMOTION_TOKENS:
        if stripped.startswith(tok):
            starts_emotion = True
            break
    for emoji in EMOTION_EMOJIS:
        if stripped.startswith(emoji):
            starts_emotion = True
            break
    results["starts_with_emotion"] = starts_emotion

    return results


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

SINGLE_TURN_TESTS = [
    {
        "name":  "Basic science question",
        "input": "What is the speed of sound?",
    },
    {
        "name":  "Biology question",
        "input": "How do cells divide?",
    },
    {
        "name":  "Jellyfish question",
        "input": "What do jellyfish eat?",
    },
    {
        "name":  "Math question",
        "input": "What is the Pythagorean theorem?",
    },
    {
        "name":  "History question",
        "input": "Who was Charles Darwin?",
    },
    {
        "name":  "Tool use — search lookup",
        "input": "look up the speed of light",
        "check_tool": True,
    },
]

MULTI_TURN_TESTS = [
    {
        "name": "Multi-turn science",
        "turns": [
            "What causes rainbows?",
            "Why do rainbows appear in an arc?",
        ],
    },
    {
        "name": "Multi-turn with follow-up",
        "turns": [
            "Explain gravity briefly.",
            "How does it relate to Einstein's general relativity?",
        ],
    },
    {
        "name": "Multi-turn jellyfish",
        "turns": [
            "Tell me about jellyfish life cycles.",
            "What stage comes before the medusa?",
        ],
    },
    {
        "name": "Humble / I don't know",
        "turns": [
            "What is the exact mass of a specific electron in the brain of a tardigrade?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Bioluminescence experiment
# ---------------------------------------------------------------------------

def check_bioluminescence_experiment(sana: SanaInference) -> dict:
    """
    Test: ask about bioluminescence.
    - Response must be ≥ 20 words (model can describe it from pretraining)
    - The word 'bioluminescence' must NOT appear in sana_dialogues.jsonl
    This verifies that knowledge from pretraining is retained through fine-tuning.
    """
    result = {
        "name":            "Bioluminescence experiment",
        "response_ok":     False,
        "not_in_sft_data": False,
        "passed":          False,
        "response":        "",
        "detail":          "",
    }

    # Check SFT data
    sft_path = Path("finetune/data/sana_dialogues.jsonl")
    if sft_path.exists():
        content = sft_path.read_text(encoding="utf-8").lower()
        result["not_in_sft_data"] = "bioluminescen" not in content
    else:
        result["not_in_sft_data"] = True  # file doesn't exist, definitely not there
        result["detail"] += "SFT data file not found. "

    # Ask the model
    sana.reset()
    response = sana.chat("What is bioluminescence?")
    result["response"] = response

    word_count = len(response.split())
    result["response_ok"] = word_count >= 20

    if not result["response_ok"]:
        result["detail"] += f"Response too short ({word_count} words). "

    if not result["not_in_sft_data"]:
        result["detail"] += "WARNING: 'bioluminescence' found in SFT data — experiment compromised. "

    result["passed"] = result["response_ok"] and result["not_in_sft_data"]
    return result


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(args):
    print("=" * 65)
    print("Sana 🪼 Fine-tune Evaluation")
    print("=" * 65)

    # Load model
    try:
        sana = SanaInference(
            checkpoint_dir     = args.checkpoint,
            max_new_tokens     = 200,
            temperature        = 0.7,
            top_p              = 0.9,
            repetition_penalty = 1.3,
        )
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    all_pass = []

    # ----- Single turn tests -----
    print("\n[SINGLE-TURN TESTS]")
    print("-" * 55)

    for test in SINGLE_TURN_TESTS:
        sana.reset()
        response = sana.chat(test["input"])
        checks = check_response(response)
        passed = all(checks.values())

        # Special check for tool use
        if test.get("check_tool"):
            has_tool = "<tool>" in response or "tool>" in response
            checks["uses_tool"] = has_tool
            passed = passed and has_tool

        all_pass.append(passed)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} | {test['name']}")
        print(f"  Input:    {test['input']!r}")
        print(f"  Response: {response[:120]!r}{'...' if len(response) > 120 else ''}")
        if not passed:
            for k, v in checks.items():
                if not v:
                    print(f"  ↳ FAILED check: {k}")

    # ----- Multi-turn tests -----
    print("\n[MULTI-TURN TESTS]")
    print("-" * 55)

    for test in MULTI_TURN_TESTS:
        sana.reset()
        turn_results = []
        print(f"\n[{test['name']}]")
        for i, turn_input in enumerate(test["turns"], 1):
            response = sana.chat(turn_input)
            checks = check_response(response)
            turn_passed = all(checks.values())
            turn_results.append(turn_passed)

            status = "✓" if turn_passed else "✗"
            print(f"  {status} Turn {i}: {turn_input!r}")
            print(f"         → {response[:100]!r}{'...' if len(response) > 100 else ''}")
            if not turn_passed:
                for k, v in checks.items():
                    if not v:
                        print(f"         ↳ FAILED: {k}")

        test_passed = all(turn_results)
        all_pass.append(test_passed)

    # ----- Bioluminescence experiment -----
    print("\n[BIOLUMINESCENCE EXPERIMENT]")
    print("-" * 55)

    bio_result = check_bioluminescence_experiment(sana)
    status = "✓ PASS" if bio_result["passed"] else "✗ FAIL"
    print(f"{status} | {bio_result['name']}")
    print(f"  Not in SFT data:  {'Yes ✓' if bio_result['not_in_sft_data'] else 'No ✗'}")
    print(f"  Response ≥20 words: {'Yes ✓' if bio_result['response_ok'] else 'No ✗'}")
    print(f"  Response preview: {bio_result['response'][:120]!r}")
    if bio_result["detail"]:
        print(f"  Detail: {bio_result['detail']}")
    all_pass.append(bio_result["passed"])

    # ----- Summary -----
    total_pass  = sum(all_pass)
    total_tests = len(all_pass)
    pct         = total_pass / total_tests * 100

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Tests passed: {total_pass}/{total_tests} ({pct:.1f}%)")

    if pct >= 70:
        print(f"✓ PASSED ({pct:.1f}% ≥ 70% threshold)")
        print("  Model shows good instruction-following and persona consistency.")
    else:
        print(f"✗ BELOW THRESHOLD ({pct:.1f}% < 70%)")
        print("  Review: check emotion token, tool call, and SFT data quality.")


def main():
    parser = argparse.ArgumentParser(description="Test Sana 🪼 fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/finetune",
                        help="Checkpoint directory")
    args = parser.parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
