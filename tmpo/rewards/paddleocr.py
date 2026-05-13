"""OCR reward model: evaluates text fidelity in generated images.

Backend: EasyOCR (PyTorch-native, works on H800 / CUDA 12.6 without any paddle dependency)

Score = 1 - NED(detected_text, target_text)
where NED = Normalized Edit Distance ∈ [0, 1].

Target text extraction priority:
  1. [OCR_TARGET: <text>] tag injected by PromptDataset  (preferred, exact match)
  2. Quoted strings in prompt, e.g. "a sign saying 'Hello'"
  3. Full prompt as fallback
"""

import re
import numpy as np
from typing import List, Optional

import torch


def _levenshtein(s1: str, s2: str) -> int:
    """O(m*n) Levenshtein edit distance (pure Python, no extra deps)."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


class PaddleOCRRewardModel:
    """OCR-based reward model (EasyOCR backend).

    Uses EasyOCR which is PyTorch-native and works on any CUDA version / GPU.
    Class name kept as PaddleOCRRewardModel for backward compatibility with
    config files and compute.py registration.

    Args:
        lang: OCR language list, e.g. ["ch_sim","en"] or ["en"]
        score_mode: scoring metric
            "ned"    – 1 - Normalized Edit Distance (default, recommended)
            "recall" – character recall (fraction of target chars found)
            "f1"     – character-level F1
    """

    _OCR_TAG_RE = re.compile(r'\[OCR_TARGET:\s*(.+?)\]', re.DOTALL)
    _QUOTE_RE   = re.compile(r'["\u201c\u201d\u2018\u2019\']([\s\S]+?)["\u201c\u201d\u2018\u2019\']')

    def __init__(
        self,
        lang: str = "ch",
        score_mode: str = "ned",
        **kwargs,
    ):
        self.score_mode = score_mode

        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError(
                "EasyOCR not installed. Run: pip install easyocr"
            ) from e

        lang_map = {
            "ch":    ["ch_sim", "en"],
            "en":    ["en"],
            "japan": ["ja", "en"],
            "korean": ["ko", "en"],
        }
        ocr_langs = lang_map.get(lang, ["en"])

        self.reader = easyocr.Reader(ocr_langs, gpu=False, verbose=False)

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _extract_target(self, prompt: str) -> str:
        """Extract target text from prompt string (three-level priority)."""
        m = self._OCR_TAG_RE.search(prompt)
        if m:
            return m.group(1).strip()
        quotes = self._QUOTE_RE.findall(prompt)
        if quotes:
            return " ".join(quotes)
        return prompt.strip()

    def _run_ocr(self, image) -> str:
        """Run EasyOCR on a PIL.Image or numpy array; return concatenated text."""
        try:
            img_np = np.array(image) if not isinstance(image, np.ndarray) else image
            results = self.reader.readtext(img_np, detail=0)  # detail=0 → list of strings
            return " ".join(results) if results else ""
        except Exception:
            return ""

    def _score(self, target: str, detected: str) -> float:
        """Compute similarity score ∈ [0, 1]."""
        target   = target.strip()
        detected = detected.strip()

        if not target:
            return 1.0 if not detected else 0.5
        if not detected:
            return 0.0

        if self.score_mode == "ned":
            ed  = _levenshtein(target, detected)
            ned = ed / max(len(target), len(detected))
            return float(max(0.0, 1.0 - ned))

        elif self.score_mode == "recall":
            t_chars = list(target.lower().replace(" ", ""))
            d_str   = detected.lower().replace(" ", "")
            if not t_chars:
                return 1.0
            return sum(1 for c in t_chars if c in d_str) / len(t_chars)

        elif self.score_mode == "f1":
            t_chars = list(target.lower().replace(" ", ""))
            d_chars = list(detected.lower().replace(" ", ""))
            if not t_chars or not d_chars:
                return 0.0
            from collections import Counter
            t_cnt, d_cnt = Counter(t_chars), Counter(d_chars)
            common = sum((t_cnt & d_cnt).values())
            precision = common / len(d_chars)
            recall    = common / len(t_chars)
            if precision + recall == 0:
                return 0.0
            return float(2 * precision * recall / (precision + recall))

        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    # ──────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        """Score images against target text extracted from prompts.

        Args:
            images:  list of PIL.Image
            prompts: list of prompt strings (may contain [OCR_TARGET: ...] tag)

        Returns:
            List of float scores ∈ [0, 1]
        """
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")

        scores = []
        for image, prompt in zip(images, prompts):
            target   = self._extract_target(prompt)
            detected = self._run_ocr(image)
            scores.append(self._score(target, detected))

        return scores
