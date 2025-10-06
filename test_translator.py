import os
import importlib


def test_import_translator_module():
    mod = importlib.import_module("translator")
    assert hasattr(mod, "translate_text")


def test_translate_en_to_hi_non_empty():
    from translator import translate_text

    out = translate_text("Hello, how are you?", "en", "hi")
    assert isinstance(out, str)
    assert out.strip() != ""


def test_translate_hi_to_en_non_empty():
    from translator import translate_text

    out = translate_text("नमस्ते, आप कैसे हैं?", "hi", "en")
    assert isinstance(out, str)
    assert out.strip() != ""


