"""Duplicate and uid audit of the raw Multi3NLU++ rows: report only, never a fix.

Normalization: NFC, then the Turkish capitals I->ı and İ->i, then lowercase, every Unicode punctuation
character (category P*) replaced by a space, whitespace collapsed. A near-duplicate is exactly an equal normalized
text; there is no fuzzy threshold. Generic case-folding is never used: plain lower() turns İ into i + combining dot,
and the Turkish mapping turns Latin acronyms into dotless forms (PIN -> pın), so acronym case variants such as
"PIN" vs "pin" are not merged. audit.md discloses this.

Policy: cross-split duplicates are reported and the splits stay as published; the flag lists in dup_flags.json are
there for a dedup sensitivity check. Identical normalized text with conflicting labels is listed with both label sets
and never fails. Duplicate uids hard-fail prep in pilot.data before this runs; the audit repeats the uid count as
evidence.

Partitions for the cross-split measure (the disjoint banking splits selection_train, val, test) and the dup-flag
pairs (val vs selection_train, test vs final_train) are passed in by pilot.data.audit_layout(), derived from
pilot.data.SPLITS, so fold boundaries live in one place. Every group and uid list is sorted so the output bytes are
deterministic. Stdlib only.

The texts this module writes into audit.json and audit.md (NORMALIZATION, render_markdown) keep their original
decision labels ("DATA-04", "D-11" to "D-14", short names for the rules above) because data/splits.json pins the audit
bytes; changing a word there means a deliberate refreeze of the splits.
"""
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

NORMALIZATION = "D-11: NFC, I->ı and İ->i, lowercase, punctuation->space, collapsed whitespace"   # pinned bytes
_TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})


def normalize_tr(text: str) -> str:
    """`text` under the Turkish normalization of the module docstring (NORMALIZATION)."""
    s = unicodedata.normalize("NFC", text).translate(_TR_UPPER).lower()
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.split())


def _exact(text: str) -> str:
    """The identity key: exact-text duplicates."""
    return text


def partition(fold: int, partitions: Iterable[tuple[str, tuple[int, ...]]]) -> str:
    """The name of the partition holding `fold`; ValueError when none does."""
    for name, folds in partitions:
        if fold in folds:
            return name
    raise ValueError("fold %r outside every audit partition" % (fold,))


def _groups(rows: list[dict[str, Any]], key: Callable[[str], str],
            partitions: Iterable[tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    """Duplicate groups of `rows` under `key` (two or more rows with one key), sorted, with their counts: rows,
    groups spanning more than one partition, groups with conflicting label sets."""
    by = {}
    for r in rows:
        by.setdefault(key(r["text"]), []).append(r)
    groups = []
    for k, members in by.items():
        if len(members) < 2:
            continue
        sets = {}
        for r in members:
            sets.setdefault(tuple(r["gold"]), []).append(r["uid"])
        groups.append({"text": k, "uids": sorted(r["uid"] for r in members),
                       "partitions": sorted({partition(r["fold"], partitions) for r in members}),
                       "label_sets": [{"labels": list(labels), "uids": sorted(uids)}
                                      for labels, uids in sorted(sets.items())]})
    groups.sort(key=lambda g: (g["text"], g["uids"][0]))
    return {"n_groups": len(groups), "n_rows": sum(len(g["uids"]) for g in groups),
            "n_cross_split_groups": sum(len(g["partitions"]) > 1 for g in groups),
            "n_conflicting_groups": sum(len(g["label_sets"]) > 1 for g in groups),   # reported, never raised
            "groups": groups}


def _flags(rows: list[dict[str, Any]], key: Callable[[str], str],
           flag_pairs: Iterable[tuple[str, tuple[int, ...], str, tuple[int, ...]]]) -> dict[str, list[str]]:
    """{"<eval>_in_<train>": uids of eval-split rows whose text occurs in the train split} for each flag pair."""
    out = {}
    for ev, ev_folds, tr, tr_folds in flag_pairs:
        seen = {key(r["text"]) for r in rows if r["fold"] in tr_folds}
        out["%s_in_%s" % (ev, tr)] = sorted(r["uid"] for r in rows if r["fold"] in ev_folds and key(r["text"]) in seen)
    return out


def build_audit(banking_rows: list[dict[str, Any]], hotel_rows: list[dict[str, Any]],
                partitions: Iterable[tuple[str, tuple[int, ...]]],
                flag_pairs: Iterable[tuple[str, tuple[int, ...], str, tuple[int, ...]]],
                flag_rows: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Raw rows ({"uid", "text", "gold", "fold", ...}) -> (audit dict, dup_flags dict). Pure; nothing is dropped.

    partitions: ((name, folds), ...) disjoint; flag_pairs: ((eval name, eval folds, train name, train folds), ...);
    both from pilot.data.audit_layout().

    The audit describes every raw row. The dup flags are computed over flag_rows (default: banking_rows), the banking
    rows actually written to the splits, so a dedup check never names a uid that is not in any split.
    """
    seen = {}
    for r in list(banking_rows) + list(hotel_rows):
        seen[r["uid"]] = seen.get(r["uid"], 0) + 1
    banking_norm = {normalize_tr(r["text"]) for r in banking_rows}
    audit = {
        "normalization": NORMALIZATION,
        "uids": {"n_rows": sum(seen.values()), "n_unique": len(seen),
                 "duplicates": sorted(u for u, n in seen.items() if n > 1)},
        "exact": _groups(banking_rows, _exact, partitions),
        "normalized": _groups(banking_rows, normalize_tr, partitions),
        "hotels_in_banking": {"uids": sorted(r["uid"] for r in hotel_rows if normalize_tr(r["text"]) in banking_norm)},
    }
    audit["hotels_in_banking"]["n"] = len(audit["hotels_in_banking"]["uids"])
    flag_rows = banking_rows if flag_rows is None else flag_rows
    flags = {"normalized": _flags(flag_rows, normalize_tr, flag_pairs), "exact": _flags(flag_rows, _exact, flag_pairs)}
    return audit, flags


def render_markdown(audit: dict[str, Any], flags: dict[str, Any]) -> str:
    """audit.md from the audit and dup_flags dicts (its bytes are pinned in data/splits.json)."""
    u, h = audit["uids"], audit["hotels_in_banking"]
    lines = [
        "# Data audit (DATA-04)",
        "",
        "Duplicates are reported only: the splits stay as published and no row is deduplicated, relabelled or "
        "dropped (D-12). Conflicting labels are listed and never fail prep (D-13).",
        "",
        "Normalization: %s. Acronym case variants are not merged under D-11: the Turkish mapping turns `PIN` into "
        "`pın`, so `PIN kodumu` and `pin kodumu` count as different texts." % audit["normalization"],
        "",
        "uids: %d rows, %d unique, %d duplicated (banking + HOTELS; duplicates hard-fail prep, D-14)."
        % (u["n_rows"], u["n_unique"], len(u["duplicates"])),
        "",
        "| Banking measure | Exact | Normalized |",
        "|---|---|---|",
    ]
    e, n = audit["exact"], audit["normalized"]
    lines += [
        "| Duplicate groups / rows | %d / %d | %d / %d |" % (e["n_groups"], e["n_rows"], n["n_groups"], n["n_rows"]),
        "| Groups spanning >1 of selection_train / val / test | %d | %d |"
        % (e["n_cross_split_groups"], n["n_cross_split_groups"]),
        "| Conflicting-label groups | %d | %d |" % (e["n_conflicting_groups"], n["n_conflicting_groups"]),
        "| Test uids whose text is in final_train | %d | %d |"
        % (len(flags["exact"]["test_in_final_train"]), len(flags["normalized"]["test_in_final_train"])),
        "| Val uids whose text is in selection_train | %d | %d |"
        % (len(flags["exact"]["val_in_selection_train"]), len(flags["normalized"]["val_in_selection_train"])),
        "",
        "HOTELS rows whose normalized text appears in banking: %d." % h["n"],
        "",
        "## Conflicting labels (normalized)",
        "",
    ]
    conflicts = [g for g in n["groups"] if len(g["label_sets"]) > 1]
    for g in conflicts:
        lines.append("- `%s`: %s" % (g["text"], " vs ".join(
            "%s [%s]" % (", ".join(s["uids"]), ", ".join(s["labels"]) or "no intent") for s in g["label_sets"])))
    if not conflicts:
        lines.append("None.")
    lines += ["", "Flag lists for a dedup sensitivity row are in dup_flags.json.", ""]
    return "\n".join(lines)
