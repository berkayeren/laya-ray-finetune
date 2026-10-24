"""Offline tiny BERT base for the BERTurk trainer and scorer tests (no download, CPU, seconds).

The layout of the pinned dbmdz/bert-base-turkish-cased snapshot (pilot.berturk.BASE_FILES): a cased WordPiece
vocab.txt (the specials, then every word, digit 0-19 and punctuation mark of the conftest synthetic texts, with the
Turkish İptal/iptal pair kept apart), tokenizer_config.json with the bytes of {"do_lower_case": false}, and a random-init
2-layer 32-wide BertModel saved with save_pretrained (config.json + model.safetensors). berturk_train.py --base DIR
adds the 48-label head itself, as it does on the pinned base.

usage: from tiny_berturk import tiny_base; base_dir = tiny_base(tmp_path / "base")
"""
import json
import os
from pathlib import Path

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]          # ids 0-4, BERT's default special tokens
WORDS = ["Kartım", "Kartımı", "kayboldu", "Oda", "ayırtmak", "istiyorum", "Evet", "evet", "İptal", "iptal", "etmek",
         "cari", "Cari", "hesap", "Hmm", "tamam", "o", "zaman", "Bir", "dakika", "lütfen", "Anladım", "sanırım", "Peki",
         "bloke", "edin", "Havlu"]
PUNCT = [".", ",", "?", "!", "'"]


def vocab() -> list[str]:
    """The tiny WordPiece vocabulary in id order: the specials, the words, the digits 0-19, the punctuation."""
    return SPECIALS + WORDS + [str(i) for i in range(20)] + PUNCT


def tiny_base(path: str | Path, seed: int = 0) -> str:
    """Write a tiny cased BERT base dir at `path`; returns str(path)."""
    import torch
    import transformers

    os.makedirs(str(path), exist_ok=True)
    words = vocab()
    with open(os.path.join(str(path), "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("".join(w + "\n" for w in words))
    with open(os.path.join(str(path), "tokenizer_config.json"), "wb") as f:
        f.write(json.dumps({"do_lower_case": False}).encode("utf-8"))
    torch.manual_seed(seed)
    model = transformers.BertModel(transformers.BertConfig(
        vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64))
    model.save_pretrained(str(path))
    return str(path)
