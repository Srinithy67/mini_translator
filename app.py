"""Streamlit UI for 🈯 Mini Language Translator.

This app demonstrates a simple Seq2Seq translation pipeline using
Helsinki-NLP MarianMT models from Hugging Face. It supports English ↔ Hindi.
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def _ensure_streamlit():
    try:
        importlib.import_module("streamlit")
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit"])


_ensure_streamlit()

import streamlit as st  # type: ignore

from translator import translate_text, SUPPORTED_LANGS


st.set_page_config(page_title="🈯 Mini Language Translator", page_icon="🈯")
st.title("🈯 Mini Language Translator")

st.markdown(
    """
This mini app uses pretrained MarianMT models to translate between English and Hindi.
Use the controls below to translate your text and to swap direction.
"""
)


def main() -> None:
    # Language selection (toggle-style via selectbox)
    direction = st.selectbox(
        "Translation direction",
        options=["English → Hindi", "Hindi → English"],
        index=0,
    )

    if direction == "English → Hindi":
        src, tgt = "en", "hi"
        prompt = "Enter English text"
    else:
        src, tgt = "hi", "en"
        prompt = "Enter Hindi text"

    text = st.text_area(prompt, height=150, placeholder="Type your sentence here…")

    if st.button("Translate"):
        with st.spinner("Translating…"):
            try:
                output = translate_text(text, src, tgt)
                if not output:
                    st.warning("Translation is empty. Try a different sentence.")
                else:
                    st.success("Translation")
                    st.write(output)
            except Exception as exc:
                st.error(f"Translation failed: {exc}")

    with st.expander("How it works (Seq2Seq pipeline)"):
        st.markdown(
            "- **Tokenization**: text → subword IDs\n"
            "- **Model**: MarianMT generates target-language tokens (beam search).\n"
            "- **Decoding**: tokens → human-readable text"
        )


if __name__ == "__main__":
    main()


