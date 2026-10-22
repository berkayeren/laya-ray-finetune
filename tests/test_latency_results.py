"""The committed latency reports (plan 07-07, EVAL-06) are the real M4 runs of the final seeds and stay consistent with
the code: reported, not gated.

- every tracked results/latency/*.json is a latency/1 report labelled "reported, not gated" of one system (laya or
  berturk) on mps or cpu, bound to the committed final manifest results/postfit/<system>-final.json (its sha256,
  is_committed true) and timing exactly that manifest's seeds in order (seed, train run id, checkpoint name and
  identity, dump run id, served T)
- the protocol is the one latency.py fixes: the pilot.latency.WARMUP_POLICY warm-up (one untimed pass over all 208
  val messages per seed before its timed loop, warmup 208), pilot.latency.THREADS torch threads, fp32 with autocast
  off on the report's device for every seed, all 208 messages of the committed val split (its sha256 and uids_sha256
  equal to data/splits.json's val entry)
- every per-seed and pooled summary recomputes from the stored times with pilot.latency.summarize, so no number can be
  edited by hand
- each report passes the repository's sidecar text and path check (no message text, no uid list, no local path)
- exactly four reports are tracked, laya-mps, laya-cpu, berturk-mps and berturk-cpu, each named after its system and
  device, and results/latency/latency.md equals pilot.latency.render_markdown of them byte for byte
- README.md holds exactly one latency block, in the Phase 7 section, equal to pilot.latency.render_markdown of the
  reports (`latency.py --render results/latency`), and the Phase 7 prose before the generated blocks says
  "reported, not gated" too
- the README's seed-order disclosure (k/208 messages on which seed 0 is slower than the mean of seeds 1 and 2, the
  mean and relative excess of seed 0 over that mean, and the exact two-sided sign test) recomputes from the committed
  times: p < 1e-7 for both Laya reports, no consistent direction (p > 0.05) for both BERTurk reports (07-REVIEW
  WR-01); every disclosed sentence is asserted whole, so a number passes only for its own system, on its own device,
  in its own direction and for its own seed, the Laya sentence is asserted from its subject through its confound
  disclaimer, the seeds are bound to 0, 1, 2, and every review mutation must fail (07-REVIEW WR-01, WR-03; 07-13
  review WR-01); CLAUDE.md's rounded Laya numbers are bound to the same times (07-REVIEW IN-02)
- pilot.latency.RULES' one warm-up line, both latency module docstrings and the README Phase 7 prose state the
  seed-order limit, and no doc or code file repeats the old position-independence claim (07-REVIEW WR-01, IN-05)
- the four reports' recorded rules differ from pilot.latency.RULES at rule 3 (the warm-up line, the text in force when
  they were timed); render_markdown marks each, and the README prose says so (07-REVIEW WR-02)
"""
import json
import math
import os
import subprocess

import pytest

from pilot import latency
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, sha256_file
from test_repo_hygiene import home_dir, local_split_texts, sidecar_problems

pytestmark = pytest.mark.unit

LATENCY = os.path.join(REPO_ROOT, "results", "latency")
DEVICES = ("mps", "cpu")
N_VAL = 208
SEED_KEYS = ("seed", "train_run_id", "checkpoint_name", "checkpoint_identity", "dump_run_id", "temperature")


def committed():
    """[(repo-relative path, parsed report)] of the tracked results/latency/*.json, in sorted order (git ls-files)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "results/latency/"], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    rels = sorted(f for f in out.stdout.split("\0") if f.endswith(".json"))
    reports = []
    for rel in rels:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            reports.append((rel, json.load(f)))
    return reports


@pytest.fixture(scope="module")
def reports():
    found = committed()
    assert found, "no latency report is tracked under results/latency/"
    return found


def test_committed_reports_bind_the_final_manifests(reports):
    for rel, r in reports:
        assert r["schema"] == latency.SCHEMA and r["label"] == latency.LABEL, rel
        assert r["system"] in latency.SYSTEMS and r["device"] in DEVICES, rel
        fm = r["final_manifest"]
        path = "results/postfit/%s-final.json" % r["system"]
        assert fm["path"] == path, rel
        assert fm["sha256"] == sha256_file(os.path.join(REPO_ROOT, path)), \
            "%s was timed against another %s than the committed one" % (rel, path)
        assert fm["is_committed"] is True, rel
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            manifest = json.load(f)
        assert [{k: s[k] for k in SEED_KEYS} for s in r["seeds"]] == \
            [{k: s[k] for k in SEED_KEYS} for s in manifest["seeds"]], \
            "%s does not time exactly the final seeds of %s in order" % (rel, path)


def test_protocol_settings(reports):
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        val = json.load(f)["splits"]["val"]
    for rel, r in reports:
        assert r.get("warmup_policy") == latency.WARMUP_POLICY, \
            "%s was timed before the %s warm-up; regenerate it with latency.py" % (rel, latency.WARMUP_POLICY)
        assert r["warmup"] == N_VAL and r["threads"] == latency.THREADS, rel
        assert r["dtype"] == "torch.float32" and r["amp_enabled"] is False, rel
        for s in r["seeds"]:
            assert s["state"] == {"device": r["device"], "dtype": "torch.float32", "amp_enabled": False}, \
                "%s seed %d" % (rel, s["seed"])
        sp = r["split"]
        assert sp["name"] == "val" and sp["n_messages"] == N_VAL, rel
        assert sp["sha256"] == val["sha256"] and sp["uids_sha256"] == val["uids_sha256"], \
            "%s was not timed on the committed val split" % rel


def test_summaries_recompute_from_times(reports):
    for rel, r in reports:
        pooled = []
        for s in r["seeds"]:
            times = s["times_s"]
            where = "%s seed %d on %s" % (r["system"], s["seed"], r["device"])
            assert len(times) == N_VAL, "%s seed %d" % (rel, s["seed"])
            assert all(isinstance(t, float) and math.isfinite(t) and t > 0 for t in times), \
                "%s seed %d" % (rel, s["seed"])
            assert s["summary"] == latency.summarize(times, where), "%s seed %d" % (rel, s["seed"])
            pooled += times
        assert r["pooled"] == latency.summarize(pooled, rel), rel
        assert r["pooled"]["n"] == N_VAL * len(r["seeds"]), rel


def test_reports_hold_no_text_or_paths(reports):
    home, texts = home_dir(), local_split_texts()
    bad = {rel: p for rel, _ in reports for p in [sidecar_problems(os.path.join(REPO_ROOT, rel), home, texts)] if p}
    assert bad == {}, "latency reports hold text, local paths or subset runs: %s" % bad


def test_exactly_four_reports(reports):
    names = [os.path.basename(rel) for rel, _ in reports]
    assert names == ["berturk-cpu.json", "berturk-mps.json", "laya-cpu.json", "laya-mps.json"], names
    for rel, r in reports:
        assert os.path.basename(rel) == "%s-%s.json" % (r["system"], r["device"]), rel


def test_markdown_matches_render():
    with open(os.path.join(LATENCY, latency.MD_NAME), encoding="utf-8") as f:
        md = f.read()
    assert md == latency.render_markdown(latency.load_reports(LATENCY)), \
        "results/latency/latency.md != render_markdown of the reports; rerun latency.py --render results/latency --out"
    assert latency.LABEL in md


RULES_DRIFT = {"berturk-cpu": [3], "berturk-mps": [3], "laya-cpu": [3], "laya-mps": [3]}
RULES_FIELD_DISCLOSURE = (
    "The `rules` field of each of the four reports keeps the rules text in force when it was timed: its warm-up line "
    "(rule 3) still says that a seed's times do not depend on when the seed runs, a claim this design cannot support "
    "(see above). The reports are the audit record and stay as timed; the rules in the generated block below are the "
    "current wording, and the block marks every report whose recorded rules differ from them.")


def test_recorded_rules_drift_is_disclosed(reports):
    """Each committed report keeps the rules text in force when it was timed, so its recorded rule 3 (the warm-up line)
    differs from pilot.latency.RULES; rules_drift finds exactly that set, render_markdown marks each report, the
    README says so, and the paraphrase is bound to the recorded text (07-REVIEW WR-02)."""
    by_name = {os.path.basename(rel)[:-len(".json")]: r for rel, r in reports}
    drift = {name: latency.rules_drift(r) for name, r in by_name.items()}
    assert drift == RULES_DRIFT, \
        "rules_drift of the committed reports is %r, not RULES_DRIFT %r; update RULES_DRIFT and the README Phase 7 " \
        "prose (07-REVIEW WR-02)" % (drift, RULES_DRIFT)
    warm = "warm-up (%s)" % latency.WARMUP_POLICY
    for name, r in by_name.items():
        assert len(r["rules"]) == len(latency.RULES), \
            "%s records %d rules, RULES holds %d" % (name, len(r["rules"]), len(latency.RULES))
        own = [i for i, (a, b) in enumerate(zip(r["rules"], latency.RULES), 1) if a != b]
        assert own == RULES_DRIFT[name], "%s: rules differ at %r, not %r" % (name, own, RULES_DRIFT[name])
        assert r["rules"][2].startswith(warm) and latency.RULES[2].startswith(warm), \
            "%s: recorded rule 3 and RULES[2] must both be the %r line" % (name, warm)
        assert OLD_CLAIMS[0] in r["rules"][2], \
            "%s: recorded rule 3 no longer holds %r, which the README paraphrases" % (name, OLD_CLAIMS[0])
    md = latency.render_markdown(latency.load_reports(LATENCY))
    assert md.count(latency.RULES_DRIFT_LINE % "3") == len(RULES_DRIFT), \
        "the render marks %d report(s) as drifting at rule 3, not %d" \
        % (md.count(latency.RULES_DRIFT_LINE % "3"), len(RULES_DRIFT))
    assert "3. %s" % warm in md, "the render does not number the rules (no %r)" % ("3. %s" % warm)
    assert RULES_FIELD_DISCLOSURE in " ".join(phase7_prose().split()), \
        "the README Phase 7 prose must state %r (07-REVIEW WR-02)" % RULES_FIELD_DISCLOSURE
    base = by_name["laya-cpu"]
    same = dict(base, rules=list(latency.RULES))
    assert latency.rules_drift(same) == [], latency.rules_drift(same)
    assert "Recorded rules" not in latency.render_markdown([same])
    short = dict(base, rules=list(latency.RULES)[:-1])
    assert latency.rules_drift(short) == [len(latency.RULES)], latency.rules_drift(short)
    bare = {k: v for k, v in base.items() if k != "rules"}
    assert latency.rules_drift(bare) == list(range(1, len(latency.RULES) + 1)), latency.rules_drift(bare)


README = os.path.join(REPO_ROOT, "README.md")
LATENCY_START, LATENCY_END = "<!-- latency:start -->", "<!-- latency:end -->"
PHASE7_HEADING = "## Confidence, out-of-domain and latency metrics (Phase 7)"


def test_readme_latency_block_matches_render():
    """The README block between the latency markers is pilot.latency.render_markdown of the committed reports; a hand
    edit or a stale block fails, and the section's prose states the label outside the generated block."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    assert text.count(LATENCY_START) == 1 and text.count(LATENCY_END) == 1, \
        "README.md must hold exactly one pair of the latency markers"
    i, j = text.index(LATENCY_START), text.index(LATENCY_END)
    assert i < j, "the latency end marker comes before the start marker"
    assert text[i + len(LATENCY_START):j].strip() == latency.render_markdown(latency.load_reports(LATENCY)).strip(), (
        "the README latency block differs from pilot.latency.render_markdown of the committed reports; "
        "regenerate it with `python latency.py --render results/latency`")
    assert text.count(PHASE7_HEADING) == 1 and text.index(PHASE7_HEADING) < i, \
        "the latency block is not in the Phase 7 section"
    assert latency.LABEL in phase7_prose(), \
        "the Phase 7 prose must say %r next to the generated latency block" % latency.LABEL


def phase7_prose():
    """The README's Phase 7 prose: from PHASE7_HEADING to the latency start marker, cut at the confidence block."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    return text[text.index(PHASE7_HEADING):text.index(LATENCY_START)].split("<!-- confidence:start -->")[0]


def sign_test_p(k, n):
    """Exact two-sided sign-test p of k successes in n untied pairs under p = 1/2 (math.comb, no scipy)."""
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def seed_order(r):
    """(k, n, mean excess ms, relative excess %, sign-test p) of seed 0 against the mean of seeds 1 and 2, message by
    message over the committed times; the report must time seeds 0, 1, 2 in that order."""
    seeds = [s["seed"] for s in r["seeds"]]
    assert seeds == [0, 1, 2], \
        "%s on %s times seeds %r; the README states that the seeds run in manifest order (0, 1, 2)" \
        % (r.get("system"), r.get("device"), seeds)
    s0, s1, s2 = (s["times_s"] for s in r["seeds"])
    d = [a - (b + c) / 2 for a, b, c in zip(s0, s1, s2)]
    k, untied = sum(x > 0 for x in d), sum(x != 0 for x in d)
    excess = 1000 * sum(d) / len(d)
    relative = 100 * (sum(s0) / sum((b + c) / 2 for b, c in zip(s1, s2)) - 1)
    return k, len(d), excess, relative, sign_test_p(k, untied)


LAYA_DISCLOSURE = (
    "The committed Laya reports show such a difference: seed 0 is slower than the mean of seeds 1 and 2 on %d/%d "
    "messages on CPU (mean %+.2f ms, %+.2f%%) and on %d/%d on MPS (mean %+.2f ms, %+.2f%%), both far beyond chance "
    "(two-sided sign test, p < 1e-7). This design cannot tell whether it comes from seed 0's weights or from running "
    "first.")
BERTURK_DISCLOSURE = (
    "BERTurk shows no consistent direction: seed 0 is slower on %d/%d messages on CPU and %d/%d on MPS (sign test p "
    "%.2f and %.2f; mean %+.2f ms, %+.2f%%, and %+.2f ms, %+.2f%%)")
ORDER_DISCLOSURE = (
    "The seeds always run in manifest order (%s), so a residual seed-order effect is not separated from a seed effect")
CLAUDE_MD_ROUNDED = "the committed Laya reports show seed 0 slower (about %.1f%% on CPU, %.1f%% on MPS)"


def assert_seed_order_disclosure(prose, by_name):
    """The README's seed-order disclosure, asserted as whole sentences rebuilt from the committed times, so a number
    passes only for its own system, on its own device, in its own direction and for its own seed; the Laya sentence is
    asserted from its subject through its confound disclaimer (07-13 review WR-01); `prose` is whitespace-collapsed."""
    lc, lm, bc, bm = (seed_order(by_name[n]) for n in ("laya-cpu", "laya-mps", "berturk-cpu", "berturk-mps"))
    for name, v in (("laya-cpu", lc), ("laya-mps", lm)):
        assert v[4] < 1e-7, "%s: the seed-0 sign test gives p %.3g, not < 1e-7; the README disclosure is stale" \
            % (name, v[4])
    for name, v in (("berturk-cpu", bc), ("berturk-mps", bm)):
        assert v[4] > 0.05, "%s: the seed-0 sign test gives p %.3g, not > 0.05; the README disclosure is stale" \
            % (name, v[4])
    orders = {", ".join(str(s["seed"]) for s in r["seeds"]) for r in by_name.values()}
    assert len(orders) == 1, "the reports time the seeds in different orders: %r" % sorted(orders)
    wants = [LAYA_DISCLOSURE % (lc[0], lc[1], lc[2], lc[3], lm[0], lm[1], lm[2], lm[3]),
             BERTURK_DISCLOSURE % (bc[0], bc[1], bm[0], bm[1], bc[4], bm[4], bc[2], bc[3], bm[2], bm[3]),
             ORDER_DISCLOSURE % orders.pop()]
    for want in wants:
        assert want in prose, "the README Phase 7 prose must state %r, recomputed from the committed times" % want


def test_readme_discloses_the_seed_order_effect(reports):
    """The README's seed-order disclosure is asserted as whole sentences rebuilt from the committed per-message times
    (the Laya seed-0 excess with k/n, mean and relative excess and p < 1e-7, from the subject naming Laya through the
    confound disclaimer (07-13 review WR-01); BERTurk's lack of a consistent direction with its counts, p, mean and
    relative excess; the manifest order), so a number cannot pass for another system, device, direction or seed
    (07-REVIEW WR-01, WR-03); CLAUDE.md's rounded Laya excess is bound too (07-REVIEW IN-02)."""
    prose = " ".join(phase7_prose().split())
    by_name = {os.path.basename(rel)[:-len(".json")]: r for rel, r in reports}
    assert_seed_order_disclosure(prose, by_name)
    with open(os.path.join(REPO_ROOT, "CLAUDE.md"), encoding="utf-8") as f:
        claude = " ".join(f.read().split())
    want = CLAUDE_MD_ROUNDED % (seed_order(by_name["laya-cpu"])[3], seed_order(by_name["laya-mps"])[3])
    assert want in claude, \
        "CLAUDE.md must state %r: its numbers must be rounded from the committed times (07-REVIEW IN-02)" % want


def swap(text, a, b):
    """text with every a and b exchanged, through a placeholder character."""
    assert "\0" not in text
    return text.replace(a, "\0").replace(b, a).replace("\0", b)


def test_seed_order_disclosure_rejects_the_review_mutations(reports):
    """The reviewers' mutations of the README prose (devices swapped, direction inverted, BERTurk counts swapped, seed
    order changed, a Laya number moved out of its sentence, the Laya numbers attributed to BERTurk, the disclosure
    turned into a denial, the confound disclaimer replaced by a causal claim) each fail assert_seed_order_disclosure;
    every surviving mutation is reported, so the check cannot quietly become presence-only again (07-REVIEW WR-01,
    WR-03; 07-13 review WR-01)."""
    good = " ".join(phase7_prose().split())
    by_name = {os.path.basename(rel)[:-len(".json")]: r for rel, r in reports}
    assert_seed_order_disclosure(good, by_name)
    bc, bm, lc = (seed_order(by_name[n]) for n in ("berturk-cpu", "berturk-mps", "laya-cpu"))
    group = "(mean %+.2f ms, %+.2f%%)" % (lc[2], lc[3])
    assert good.count(group) == 1, "the Laya CPU group %r is not in the README prose once" % group
    mutations = {
        "devices swapped": swap(good, "CPU", "MPS"),
        "direction inverted": good.replace("seed 0 is slower", "seed 0 is faster"),
        "BERTurk counts swapped": swap(good, "%d/%d" % bc[:2], "%d/%d" % bm[:2]),
        "seed order changed": good.replace("(0, 1, 2)", "(0, 2, 1)"),
        "Laya CPU number moved": good.replace(group + " ", "") + " " + group,
        "system swapped": good.replace("The committed Laya reports", "The committed BERTurk reports"),
        "disclosure denied": good.replace("The committed Laya reports show such a difference:",
                                          "Neither model shows such a difference; were it otherwise,"),
        "confound claimed": good.replace(
            "This design cannot tell whether it comes from seed 0's weights or from running first.",
            "This difference comes from seed 0's weights, not from running first."),
    }
    survivors = []
    for name, mutated in mutations.items():
        assert mutated != good, "mutation %r no longer matches the README wording; update it" % name
        try:
            assert_seed_order_disclosure(mutated, by_name)
        except AssertionError:
            continue
        survivors.append(name)
    assert survivors == [], \
        "these mutations of the README prose still pass assert_seed_order_disclosure, so the check does not bind " \
        "what they change: %r (07-13 review WR-01)" % survivors


SEED_ORDER_LIMIT = "a residual seed-order effect is not separated from a seed effect"
OLD_CLAIMS = ("depends on its position", "independent of its position", "those included",
              "first-occurrence costs are paid", "shows none")


def test_latency_wording_states_the_seed_order_limit():
    """RULES has exactly one warm-up line, and it states only what full_pass_per_seed guarantees (the same sequence per
    seed, every timed input run once before) and the seed-order limit; both module docstrings and the README Phase 7
    prose say the same, and no doc or code file repeats the old position-independence claim (07-REVIEW WR-01, IN-05,
    G-07-UAT2-1). The committed reports keep the rules text in force when they were timed, so they are not checked."""
    warm = [rule for rule in latency.RULES if rule.startswith("warm-up")]
    assert len(warm) == 1, "pilot.latency.RULES must hold exactly one warm-up line, not %d" % len(warm)
    for want in (latency.WARMUP_POLICY, "every seed follows the same sequence",
                 "every input it times has already run once on its model in this process", SEED_ORDER_LIMIT):
        assert want in warm[0], "the RULES warm-up line must say %r" % want
    texts = {"pilot/latency.py docstring": latency.__doc__,
             "README.md Phase 7 prose": phase7_prose()}
    with open(os.path.join(REPO_ROOT, "latency.py"), encoding="utf-8") as f:
        texts["latency.py"] = f.read()
    for name, text in texts.items():
        assert SEED_ORDER_LIMIT in " ".join(text.split()), "%s must say %r" % (name, SEED_ORDER_LIMIT)
    for rel in ("pilot/latency.py", "latency.py", "README.md", "CLAUDE.md", ".claude/CLAUDE.md",
                "results/latency/latency.md"):
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = " ".join(f.read().split())
        for claim in OLD_CLAIMS:
            assert claim not in text, "%s still says %r (07-REVIEW WR-01, IN-05)" % (rel, claim)
