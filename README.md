# 🈯 Mini Language Translator

## Objective
Build a mini language translator that converts text between English and Hindi using pretrained MarianMT models.  
The goal is to demonstrate a compact and efficient Seq2Seq translation pipeline suitable for small datasets.

---

## Tech Stack
- **Python 3.10+**
- **Hugging Face Transformers (MarianMT)**
- **Streamlit** (for UI)
- **pandas** (for CSV handling)
- **pytest** (for tests)

Translate between English ↔ Hindi using pretrained Helsinki-NLP MarianMT models from Hugging Face.

## Overview

This mini app demonstrates a compact text translation pipeline:

1. Tokenization — Convert input text into subword token IDs using a tokenizer.
2. Seq2Seq Generation — A MarianMT encoder–decoder model generates target-language tokens.
3. Decoding — Convert generated token IDs back to human-readable text.

We use these models:
- English → Hindi: `Helsinki-NLP/opus-mt-en-hi`
- Hindi → English: `Helsinki-NLP/opus-mt-hi-en`

The app auto-installs missing dependencies the first time you run it.

## How Seq2Seq models work (simple explanation)

Sequence-to-Sequence (Seq2Seq) models read an input sequence (source sentence) and produce an output sequence (translated sentence):

- The encoder reads the source tokens and produces hidden representations.
- The decoder generates the target tokens step by step, attending to encoder outputs.
- Beam search helps pick fluent, likely translations.

MarianMT is a family of such models released by Helsinki-NLP, trained on many language pairs.

## Project structure

```
mini_translator/
├── app.py
├── translator.py
├── data.csv
├── requirements.txt
└── README.md
```

## Setup and Run

1. (Optional) Create a virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

The first run may download the pretrained models (hundreds of MB) from Hugging Face.

## Sample data

`data.csv` contains 30–50 English↔Hindi example pairs you can use for demos.

## Tests

There are a few lightweight tests in `test_translator.py` to verify that translations are non-empty for both directions. Run them with:

```bash
python -m pytest -q
```


