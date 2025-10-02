# src/summarizer.py
import os
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Cache for models
_cached = {}

def _device_index(use_gpu: bool = True) -> int:
    """Return 0 if GPU available, else -1 (CPU)."""
    if use_gpu:
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    return -1

def load_model_and_tokenizer(model_source: str, use_gpu: bool = True, use_auth_token: str = None):
    """Load model + tokenizer once and cache them."""
    device = _device_index(use_gpu)
    key = f"{model_source}|{device}"
    if key in _cached:
        return _cached[key]

    tokenizer = AutoTokenizer.from_pretrained(model_source, token=use_auth_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, token=use_auth_token)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

    _cached[key] = (tokenizer, summarizer)
    return tokenizer, summarizer

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int = 900, overlap: int = 50) -> List[str]:
    """Split long text into overlapping chunks by token length."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    if n <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def summarize_long_text(
    text: str,
    model_source: str,
    use_gpu: bool = True,
    min_length: int = 30,
    max_length: int = 200,
    second_pass: bool = True,
    use_auth_token: str = None
) -> str:
    """Summarize long clinical text using a seq2seq model."""
    if not text or not text.strip():
        return ""

    tokenizer, summarizer = load_model_and_tokenizer(
        model_source, use_gpu=use_gpu, use_auth_token=use_auth_token
    )

    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=900, overlap=50)

    partials = []
    for ch in chunks:
        out = summarizer(
            ch,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        partials.append(out[0]["summary_text"].strip())

    combined = " ".join(partials)

    if second_pass and len(combined.split()) > max_length * 2:
        out2 = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)
        return out2[0]["summary_text"].strip()

    return combined
