"""translator.py

A lightweight translation module for the Mini Language Translator app.

Features:
- Auto-installs required packages (transformers, torch) if missing.
- Caches loaded models/tokenizers to avoid repeated downloads.
- Provides a single translate_text(text, src_lang, tgt_lang) function.

We choose Hindi (language code: "hi") for the demo. The corresponding
Hugging Face models are:
- English -> Hindi:  Helsinki-NLP/opus-mt-en-hi
- Hindi -> English:  Helsinki-NLP/opus-mt-hi-en

Both are MarianMT (Seq2Seq) models. The translation pipeline performs:
1) Tokenization: convert text to subword tokens
2) Model inference (Seq2Seq generation): generate target-language tokens
3) Decoding: convert tokens back to text
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from functools import lru_cache
from typing import Dict, Tuple


def _ensure_package(pkg_name: str, import_name: str | None = None) -> None:
    """Ensure a Python package is installed; install quietly if missing.

    Args:
        pkg_name: Name to install via pip.
        import_name: Optional different module name to import.
    """
    module_name = import_name or pkg_name
    try:
        importlib.import_module(module_name)
    except Exception:  # ImportError or other issues
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg_name]
        )


def _ensure_dependencies() -> None:
    """Install required third-party dependencies if they are missing.

    We keep this narrow to model/runtime needs; Streamlit is imported in app.py.
    """
    _ensure_package("torch")
    _ensure_package("transformers")


_ensure_dependencies()

# Now safe to import transformers
from transformers import MarianMTModel, MarianTokenizer  # type: ignore


# Supported language codes for this mini app
SUPPORTED_LANGS = {"en", "hi"}


def _normalize_lang(lang: str) -> str:
    """Normalize a language code to our supported set (e.g., EN -> en)."""
    code = (lang or "").strip().lower()
    if code not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported language code '{lang}'. Supported: {sorted(SUPPORTED_LANGS)}"
        )
    return code


def _get_model_name(src_lang: str, tgt_lang: str) -> str:
    """Return the appropriate Helsinki-NLP model name for the language pair."""
    src = _normalize_lang(src_lang)
    tgt = _normalize_lang(tgt_lang)
    if src == tgt:
        raise ValueError("Source and target languages must be different.")
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(src_lang: str, tgt_lang: str) -> Tuple[MarianMTModel, MarianTokenizer]:
    """Load and cache the MarianMT model and tokenizer for the language pair.

    Caching avoids repeated downloads and speeds up subsequent calls.
    """
    model_name = _get_model_name(src_lang, tgt_lang)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


def translate_text(text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 128) -> str:
    """Translate input text from src_lang to tgt_lang using MarianMT.

    Args:
        text: The input sentence to translate.
        src_lang: Source language code ("en" or "hi").
        tgt_lang: Target language code ("en" or "hi").
        max_new_tokens: Generation limit to keep outputs bounded.

    Returns:
        The translated string.
    """
    if not text or not text.strip():
        return ""

    model, tokenizer = _load_model_and_tokenizer(src_lang, tgt_lang)

    # Tokenize the input text -> input IDs suitable for the model
    encoded = tokenizer([text], return_tensors="pt", padding=True)

    # Generate translated tokens using the Seq2Seq model
    generated_tokens = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )

    # Decode tokens back to a string in the target language
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0] if translated else ""


__all__ = ["translate_text", "SUPPORTED_LANGS"]


