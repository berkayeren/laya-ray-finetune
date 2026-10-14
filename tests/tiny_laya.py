"""Offline tiny Laya checkpoint for the adapter contract, scorer and smoke tests (no download, CPU, seconds).

A WordLevel tokenizer over the words of the 48 intent questions plus a few synthetic Turkish words (anything else maps
to [UNK]), a 2-layer 64-wide ModernBERT encoder and a Laya head, saved through pilot.laya_adapter in the layout the
public laya.load reads. Laya itself is reached only through the adapter.

usage: from tiny_laya import tiny_checkpoint; agent_dir = tiny_checkpoint(tmp_path / "ckpt", temperature=1.3)
"""
import re

from intents import INTENTS
from pilot import laya_adapter as la

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]          # ids 0-4
EXTRA_WORDS = ["Kartım", "kayboldu", "Evet", "."]


def _vocab():
    words = set(EXTRA_WORDS)
    for text in INTENTS.values():
        words.update(re.findall(r"\w+|[^\w\s]+", text))            # the Whitespace pre-tokenizer's split
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    return vocab


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(_vocab(), unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                   mask_token="[MASK]", unk_token="[UNK]")


def tiny_checkpoint(path, temperature=1.0, seed=0):
    """Save a random-init tiny Laya checkpoint to `path` (noul temperature `temperature`); returns str(path)."""
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(seed)
    tok = tiny_tokenizer()
    cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                           num_attention_heads=4, global_attn_every_n_layers=3, local_attention=16, pad_token_id=0,
                           bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    model = la.decision_model(ModernBertModel(cfg), 2, 2)
    # "encoder" and "head_layers" are required by laya.load's compatibility check; one act_costs entry gives n_act 2
    laya_cfg = {"encoder": "tiny-modernbert-offline", "head_layers": 2, "act_costs": {"escalate": 0.5},
                "temperature": [1.0, 1.0, float(temperature)], "max_len": 64, "head_max_len": 32}
    la.save_checkpoint(model, tok, laya_cfg, str(path))
    return str(path)
