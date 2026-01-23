"""
tokenizer/tokenizer.py — Pure Python BPE tokenizer for the model.
Zero external dependencies; uses only the Python standard library (json, re, typing).
"""

import json
import re
from typing import Dict, List, Optional, Tuple


class Tokenizer:

    PAD_ID = 0
    USER_ID = 1
    SANA_ID = 2
    END_ID = 3
    TOOL_OPEN_ID = 4
    TOOL_CLOSE_ID = 5
    RESULT_OPEN_ID = 6
    RESULT_CLOSE_ID = 7
    SALUTE_ID = 8
    HAPPY_ID = 9
    THINK_ID = 10
    SAD_ID = 11
    SMUG_ID = 12
    SANA_NAME_ID = 13

    ALL_SPECIAL_IDS = set(range(14))
    STRUCTURAL_IDS = {0, 1, 2, 3}
    EMOTION_TOKEN_IDS = {8, 9, 10, 11, 12}

    _ID_TO_SPECIAL = {
        0: "<|pad|>",
        1: "<|user|>",
        2: "<|sana|>",
        3: "<|end|>",
        4: "<tool>",
        5: "</tool>",
        6: "<tool_result>",
        7: "</tool_result>",
        8: "<sana_salute>",
        9: "<sana_happy>",
        10: "<sana_think>",
        11: "<sana_sad>",
        12: "<sana_smug>",
        13: "Sana",
    }

    _SPECIAL_TO_ID = {v: k for k, v in _ID_TO_SPECIAL.items()}

    _SPECIAL_TOKEN_RE = re.compile(
        r"(<\|pad\|>|<\|user\|>|<\|sana\|>|<\|end\|>"
        r"|<tool>|</tool>|<tool_result>|</tool_result>"
        r"|<sana_salute>|<sana_happy>|<sana_think>|<sana_sad>|<sana_smug>"
        r"|\bSana\b)"
    )

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size: int = data["vocab_size"]

        self.token_to_id: Dict[str, int] = data["vocab"]

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.merges: List[Tuple[str, str]] = [tuple(m) for m in data["merges"]]
        self.merge_rank: Dict[Tuple[str, str], int] = {
            (a, b): i for i, (a, b) in enumerate(self.merges)
        }

    def _pretokenise(self, text: str) -> List[List[str]]:
        """
        Split text into words. Each word becomes a list of characters/bytes.
        Non-ASCII bytes are represented as 2-hex-char strings (e.g. 'c3').
        """

        words = re.findall(r"\w+|[^\w\s]|\s+", text)
        result = []
        for word in words:
            chars = []
            for ch in word:
                if ord(ch) < 128:
                    chars.append(ch)
                else:

                    for byte in ch.encode("utf-8"):
                        chars.append(f"{byte:02x}")
            if chars:
                result.append(chars)
        return result

    def _bpe_word(self, chars: List[str]) -> List[str]:
        """Apply BPE merges to a single word represented as a list of chars."""
        if len(chars) == 1:
            return chars

        tokens = list(chars)

        while True:

            best_rank = None
            best_i = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i

            if best_i == -1:
                break

            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        Special token strings embedded in text are detected and emitted
        with their fixed IDs — they are NOT passed through BPE.
        encode("hello") must NOT contain END_ID.
        """
        ids: List[int] = []

        parts = self._SPECIAL_TOKEN_RE.split(text)

        for part in parts:
            if not part:
                continue

            if part in self._SPECIAL_TO_ID:
                ids.append(self._SPECIAL_TO_ID[part])
            else:

                for word_chars in self._pretokenise(part):
                    bpe_tokens = self._bpe_word(word_chars)
                    for tok in bpe_tokens:
                        if tok in self.token_to_id:
                            ids.append(self.token_to_id[tok])
                        else:

                            for byte in tok.encode("utf-8"):
                                hex_str = f"{byte:02x}"
                                if hex_str in self.token_to_id:
                                    ids.append(self.token_to_id[hex_str])

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        If skip_special=True, structural special tokens (pad, user, sana, end) are omitted.
        Emotion tokens are kept as their string forms by default.
        """
        parts = []
        hex_buf = []

        def flush_hex():
            if hex_buf:
                try:
                    parts.append(bytes(hex_buf).decode("utf-8", errors="replace"))
                except Exception:
                    parts.extend(f"{b:02x}" for b in hex_buf)
                hex_buf.clear()

        for token_id in ids:
            if token_id in self.ALL_SPECIAL_IDS:
                flush_hex()
                if skip_special and token_id in self.STRUCTURAL_IDS:
                    continue
                parts.append(self._ID_TO_SPECIAL[token_id])
            elif token_id in self.id_to_token:
                tok = self.id_to_token[token_id]
                if re.fullmatch(r"[0-9a-f]{2}", tok):
                    hex_buf.append(int(tok, 16))
                else:
                    flush_hex()
                    parts.append(tok)
            else:
                flush_hex()

        flush_hex()
        return "".join(parts)

    def encode_tool_call(self, call_str: str, result_str: str) -> List[int]:
        """
        Encode a tool call + result block:
        <tool>call_str</tool><tool_result>result_str</tool_result>
        """
        ids = [self.TOOL_OPEN_ID]
        ids.extend(self.encode(call_str))
        ids.append(self.TOOL_CLOSE_ID)
        ids.append(self.RESULT_OPEN_ID)
        ids.extend(self.encode(result_str))
        ids.append(self.RESULT_CLOSE_ID)
        return ids

    def build_inference_prompt(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        max_seq_len: int = 512,
    ) -> List[int]:
        """
        Build a token-budget-aware prompt for inference.
        Format: [SANA_ID, USER_ID, ...msg..., END_ID, SANA_ID]
        History is trimmed oldest-first to fit within max_seq_len.

        history: list of (user_text, model_reply_text) tuples, oldest first
        """

        current_turn = (
            [self.USER_ID] + self.encode(message) + [self.END_ID] + [self.SANA_ID]
        )

        budget = max_seq_len - 1 - len(current_turn)

        history_tokens: List[List[int]] = []
        if history:
            for user_txt, sana_txt in history:
                turn_tokens = (
                    [self.USER_ID]
                    + self.encode(user_txt)
                    + [self.END_ID]
                    + [self.SANA_ID]
                    + self.encode(sana_txt)
                    + [self.END_ID]
                )
                history_tokens.append(turn_tokens)

        while history_tokens:
            total_hist = sum(len(t) for t in history_tokens)
            if total_hist <= budget:
                break
            history_tokens.pop(0)

        ids = [self.SANA_ID]
        for turn in history_tokens:
            ids.extend(turn)
        ids.extend(current_turn)

        return ids[:max_seq_len]

    def build_multiturn_training_sequence(
        self,
        conversations: List[Dict],
        max_seq_len: int = 512,
        train_tool_results: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Build (input_ids, label_ids) for multi-turn SFT training.
        Conversations is a list of {"role": "user"|"sana", "content": "..."}
        (the "sana" role denotes the model's own turns).

        Label masking rules:
        - All user turn tokens                          → -100 (no loss)
        - SANA_ID role marker tokens                    → -100 (structural)
        - Model response tokens                         → active loss
        - <tool>...</tool> tokens inside a model turn   → active loss (model learns to call tools)
        - <tool_result>...</tool_result> tokens         → active loss if train_tool_results=True
          This teaches the model to predict/hallucinate plausible tool results when
          a real tool is unavailable, so inference degrades gracefully.
        """
        import re as _re

        raw_ids: List[int] = []
        is_active: List[bool] = []

        raw_ids.append(self.SANA_ID)
        is_active.append(False)

        for turn in conversations:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                turn_tokens = [self.USER_ID] + self.encode(content) + [self.END_ID]
                raw_ids.extend(turn_tokens)
                is_active.extend([False] * len(turn_tokens))
                raw_ids.append(self.SANA_ID)
                is_active.append(False)

            elif role == "sana":

                remaining = content
                seg_tokens: List[int] = []
                seg_active: List[bool] = []

                pattern = _re.compile(
                    r"(<tool>.*?</tool>)|(<tool_result>.*?</tool_result>)", _re.DOTALL
                )
                last = 0
                for m in pattern.finditer(remaining):

                    plain = remaining[last : m.start()]
                    if plain:
                        toks = self.encode(plain)
                        seg_tokens.extend(toks)
                        seg_active.extend([True] * len(toks))

                    matched = m.group(0)
                    if matched.startswith("<tool>"):

                        toks = self.encode(matched)
                        seg_tokens.extend(toks)
                        seg_active.extend([True] * len(toks))
                    elif matched.startswith("<tool_result>"):

                        toks = self.encode(matched)
                        seg_tokens.extend(toks)
                        seg_active.extend([train_tool_results] * len(toks))

                    last = m.end()

                plain = remaining[last:]
                if plain:
                    toks = self.encode(plain)
                    seg_tokens.extend(toks)
                    seg_active.extend([True] * len(toks))

                seg_tokens.append(self.END_ID)
                seg_active.append(True)

                raw_ids.extend(seg_tokens)
                is_active.extend(seg_active)

        raw_ids = raw_ids[: max_seq_len + 1]
        is_active = is_active[: max_seq_len + 1]

        input_ids = raw_ids[:-1]
        label_ids = [
            raw_ids[i + 1] if is_active[i + 1] else -100 for i in range(len(input_ids))
        ]

        pad_len = max_seq_len - len(input_ids)
        input_ids = input_ids + [self.PAD_ID] * pad_len
        label_ids = label_ids + [-100] * pad_len

        return input_ids, label_ids

    def __len__(self) -> int:
        return self.vocab_size
