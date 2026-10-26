"""The demo UI (ui/, Vite + React + TypeScript): the built bundle served by demo.create_app on 127.0.0.1, the npm
outputs gitignored, and the static guards every later UI plan runs: exact npm pins, the committed lock (registry
tarballs with sha512 integrity, no install script), ui/.npmrc's install policy, tokens.css equal to the UI-SPEC, and
no verdict word, question text, browser storage, cookie, inner HTML, case-changing CSS or external URL in ui/src or
ui/index.html (comments included), and no ui/src path the root .gitignore would swallow.

The chat contracts: the six example messages of ui/src/examples.ts are demo-authored and never the message text of a
split file (test_examples_are_not_dataset_text, data marker: it compares them with every local data/splits/*.jsonl
after normalization and never prints a split text), together they cover the Turkish letters, and the chat copy of the
UI-SPEC appears in ui/src exactly. The dashboard contracts: the dashboard copy of the UI-SPEC appears in ui/src exactly
(test_dashboard_copy_contract), and the routes are #/chat and the four #/dashboard/ views in the UI-SPEC's order, each
with a component in Dashboard.tsx's VIEW_COMPONENTS (test_dashboard_routes).

test_built_bundle_is_served needs `npm --prefix ui ci && npm --prefix ui run build` first (it skips, naming that
command, while ui/dist/index.html is absent); test_examples_are_not_dataset_text needs `make data` (it skips, naming
it, while data/splits holds no *.jsonl); every other test reads only tracked files. Runs in `make test` (the data test
under `make test-all` or `pytest -m data`).
"""
import json
import os
import re
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import demo
import intents
import pilot.data
from pilot import verdict

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
UI = os.path.join(REPO, "ui")
BUILD = "npm --prefix ui ci && npm --prefix ui run build"


def ignored(rel: str) -> bool:
    """True when `git check-ignore --no-index` ignores the repo-relative path rel."""
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", "--", rel], cwd=REPO).returncode == 0


def test_built_bundle_is_served() -> None:
    if not os.path.isfile(os.path.join(demo.UI_DIST, "index.html")):
        pytest.skip("ui/dist/index.html is absent; build the UI first: %s" % BUILD)
    from starlette.testclient import TestClient
    with ThreadPoolExecutor(max_workers=1) as ex:
        app = demo.create_app(demo.new_state("cpu"), ex, {}, demo.UI_DIST)
        with TestClient(app, base_url="http://127.0.0.1") as c:
            r = c.get("/")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/html")
            assert 'id="root"' in r.text
            assets = sorted(set(re.findall(r'(?:src|href)="(/assets/[^"]+)"', r.text)))
            assert assets, "the served index.html references no /assets/ file"
            for path in assets:
                a = c.get(path)
                assert a.status_code == 200, "%s answered %d" % (path, a.status_code)
            s = c.get("/api/status")
            assert s.status_code == 200
            assert s.json()["chat"] == "loading"


def test_ui_outputs_are_gitignored() -> None:
    for rel in ("ui/node_modules/x/index.js", "ui/dist/index.html"):
        assert ignored(rel), "%s must be gitignored" % rel
    for rel in ("ui/src/App.tsx", "ui/package-lock.json"):
        assert not ignored(rel), "%s must not be gitignored" % rel


# --- static guards over the tracked UI sources (every later UI plan runs them) -----------------------------------

DEPENDENCIES = {"react": "19.3.0", "react-dom": "19.3.0", "react-is": "19.3.0", "recharts": "3.10.1"}
DEV_DEPENDENCIES = {"vite": "8.3.4", "@vitejs/plugin-react": "6.1.2", "typescript": "6.0.3", "@types/react": "19.3.0",
                    "@types/react-dom": "19.3.0"}
EXACT = re.compile(r"^\d+\.\d+\.\d+$")
REGISTRY = "https://registry.npmjs.org/"
# The one locked package the registry marks hasInstallScript: fsevents 2.3.3, vite's optional, macOS-only file-watcher
# dependency. Its registry metadata carries an implicit `install: node-gyp rebuild` (gypfile true), yet its tarball holds
# no binding.gyp, only the prebuilt fsevents.node, and ui/.npmrc's ignore-scripts=true runs no lifecycle script at all.
# It is allowed at exactly this version, integrity and shape; any other package with an install script fails.
INSTALL_SCRIPT_ALLOWED = {
    "node_modules/fsevents": {
        "version": "2.3.3", "dev": True, "optional": True, "os": ["darwin"],
        "integrity": "sha512-5xoDfX+fL7faATnagmWPpbFtwh/R77WmMMqqHGS65C3vvB0YHrgF+B1YmZ3441tMj5n63k0212XNoJwzlhffQw==",
    },
}
# 11-UI-SPEC.md: Spacing Scale, Typography, Color and the fixed sizes, as CSS custom properties of tokens.css
TOKENS = {
    "space-xs": "4px", "space-sm": "8px", "space-md": "16px", "space-lg": "24px", "space-xl": "32px",
    "space-2xl": "48px", "space-3xl": "64px",
    "text-label": "14px", "text-body": "16px", "text-heading": "20px", "text-display": "28px",
    "weight-regular": "400", "weight-semibold": "600",
    "leading-label": "1.4", "leading-body": "1.5", "leading-heading": "1.25", "leading-display": "1.2",
    "font-sans": 'system-ui, -apple-system, "Segoe UI", sans-serif',
    "font-mono": 'ui-monospace, "SF Mono", Menlo, monospace',
    "color-dominant": "#f9f9f7", "color-surface": "#ffffff", "color-accent": "#0f766e",
    "color-accent-hover": "#0d6660", "color-disabled": "#c3c2b7", "color-destructive": "#d03b3b",
    "ink-primary": "#0b0b0b", "ink-secondary": "#52514e", "ink-muted": "#6b6a66",
    "color-hairline": "#e1e0d9", "color-axis": "#c3c2b7", "color-user-bubble": "#f0efec", "color-code": "#f6f5f2",
    "series-laya": "#2a78d6", "series-berturk": "#eb6834", "series-superseded": "#898781",
    "series-reference": "#52514e",
    "tone-positive-text": "#006300", "tone-positive-fill": "#e3f4e3", "tone-positive-border": "#0ca30c",
    "tone-negative-text": "#b42318", "tone-negative-fill": "#fdeaea", "tone-negative-border": "#d03b3b",
    "tone-caution-text": "#7a4d00", "tone-caution-fill": "#fdf0d5", "tone-caution-border": "#fab219",
    "tone-neutral-text": "#52514e", "tone-neutral-fill": "#f0efec", "tone-neutral-border": "#c3c2b7",
    "header-height": "56px", "sidebar-width": "240px", "chat-max": "1024px", "content-max": "1200px",
    "control-min": "44px", "chip-height": "28px", "badge-height": "24px", "bar-height": "8px", "swatch": "12px",
    "chart-height": "280px", "chart-height-sm": "240px",
    "radius-card": "8px", "radius-bubble": "16px", "radius-chip": "14px", "radius-badge": "4px",
}
TOKEN_RE = re.compile(r"--([a-z0-9-]+)\s*:\s*([^;]+?)\s*;")
FORBIDDEN_APIS = ("dangerouslySetInnerHTML", "innerHTML", "localStorage", "sessionStorage", "document.cookie",
                  "text-transform")
URL_RE = re.compile(r"https?://[^\s\"'`<>)]*")
SVG_NS = "http://www.w3.org/2000/svg"
WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")
IGNORED_NAMES = ("lib", "build", "dist", "parts", "var", "share", "downloads", "target", "instance", "env")


def tracked(*paths: str) -> list[str]:
    """The tracked repo-relative files under the given paths (git ls-files)."""
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.split()


def read_json(rel: str) -> dict[str, Any]:
    """The parsed JSON of the repo-relative file rel."""
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


def ui_sources() -> list[str]:
    """Every file under ui/src (os.walk, sorted) plus ui/index.html, repo-relative."""
    out = ["ui/index.html"]
    for root, dirs, files in os.walk(os.path.join(UI, "src")):
        dirs.sort()
        out += [os.path.relpath(os.path.join(root, f), REPO) for f in sorted(files)]
    return out


def test_ui_pins_are_exact() -> None:
    pkg = read_json("ui/package.json")
    assert pkg["dependencies"] == DEPENDENCIES
    assert pkg["devDependencies"] == DEV_DEPENDENCIES
    for name, version in {**pkg["dependencies"], **pkg["devDependencies"]}.items():
        assert EXACT.match(version), "%s is pinned as %r, not an exact version" % (name, version)
    assert pkg["engines"] == {"node": ">=22.12.0"}


def test_ui_lockfile_is_committed_and_clean() -> None:
    assert tracked("ui/package-lock.json") == ["ui/package-lock.json"], "ui/package-lock.json must be tracked"
    lock, pkg = read_json("ui/package-lock.json"), read_json("ui/package.json")
    assert lock["lockfileVersion"] == 3
    root = lock["packages"][""]
    assert root.get("dependencies") == pkg["dependencies"]
    assert root.get("devDependencies") == pkg["devDependencies"]
    problems = []
    for key, entry in sorted(lock["packages"].items()):
        if key == "":
            continue
        if not entry.get("resolved", "").startswith(REGISTRY):
            problems.append("%s: resolved %r is not under %s" % (key, entry.get("resolved"), REGISTRY))
        if not entry.get("integrity", "").startswith("sha512-"):
            problems.append("%s: integrity %r is not sha512" % (key, entry.get("integrity")))
        if entry.get("hasInstallScript"):
            allowed = INSTALL_SCRIPT_ALLOWED.get(key)
            if allowed is None or any(entry.get(k) != v for k, v in allowed.items()):
                problems.append("%s: declares an install script" % key)
    assert problems == [], problems
    home = os.path.expanduser("~").rstrip("/\\")
    with open(os.path.join(UI, "package-lock.json"), encoding="utf-8") as f:
        raw = f.read()
    assert len(home) < 2 or home not in raw, "ui/package-lock.json holds the local home directory path"


def test_ui_npmrc() -> None:
    assert tracked("ui/.npmrc") == ["ui/.npmrc"], "ui/.npmrc must be tracked"
    with open(os.path.join(UI, ".npmrc"), encoding="utf-8") as f:
        lines = {ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(("#", ";"))}
    for setting in ("save-exact=true", "engine-strict=true", "ignore-scripts=true"):
        assert setting in lines, "ui/.npmrc must set %s" % setting


def test_ui_tokens_match_spec() -> None:
    with open(os.path.join(UI, "src", "styles", "tokens.css"), encoding="utf-8") as f:
        css = f.read()
    declared: dict[str, list[str]] = {}
    for name, value in TOKEN_RE.findall(css):
        declared.setdefault(name, []).append(value)
    problems = ["--%s: %s, spec %s" % (k, declared.get(k), v) for k, v in TOKENS.items() if declared.get(k) != [v]]
    assert problems == [], "tokens.css differs from the UI-SPEC: %s" % problems


def test_ui_source_has_no_verdict_or_question_literals() -> None:
    words = {verdict.INCONCLUSIVE, verdict.GO, verdict.NO_GO}
    phrases = [verdict.INCONCLUSIVE_MEANS, *intents.INTENTS.values(), *FORBIDDEN_APIS]
    problems = []
    for rel in ui_sources():
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            text = f.read()
        problems += ["%s: verdict word %r" % (rel, w) for w in sorted(words & set(WORD_RE.findall(text)))]
        problems += ["%s: %r" % (rel, p) for p in phrases if p in text]
        problems += ["%s: external URL %r" % (rel, u) for u in URL_RE.findall(text) if u != SVG_NS]
    assert problems == [], problems


def test_ui_folder_names_are_not_ignored() -> None:
    src = ui_sources()[1:]
    assert src, "ui/src holds no file"
    problems = []
    for rel in src:
        parts = os.path.relpath(rel, "ui/src").split(os.sep)
        problems += ["%s: path component %r" % (rel, p) for p in parts if p in IGNORED_NAMES]
        if ignored(rel):
            problems.append("%s is gitignored" % rel)
    assert problems == [], problems


# --- the chat: example messages and copy -------------------------------------------------------------------------

EXAMPLES_TS = os.path.join(UI, "src", "examples.ts")
DQ_RE = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
TURKISH_LETTERS = ("ğ", "ü", "ş", "ı", "ö", "ç", "İ")
# 11-UI-SPEC.md Copywriting Contract: the chat copy, each exactly as written there
CHAT_COPY = (
    "Classify message",
    "Classify a Turkish banking message",
    "Each message is classified on its own: the models see no earlier messages. "
    "This local demo stores and logs nothing.",
    "Type a Turkish banking message",
    "Try an example",
    "Laya and BERTurk are classifying…",
    "Loading the six final checkpoints…",
    "No intent",
    "No tag reached its threshold θ for this message.",
    "Low means route to a human.",
    "Chat unavailable",
    "The dashboard still works.",
    "Open the dashboard",
    "Clear messages",
    "Keep messages",
    "Remove every message from this tab? Nothing is stored, so they cannot be restored.",
    "Show all 48 tags",
)


def example_strings() -> list[str]:
    """The double-quoted strings of ui/src/examples.ts (the file holds exactly the example messages)."""
    with open(EXAMPLES_TS, encoding="utf-8") as f:
        return DQ_RE.findall(f.read())


def normalize(text: str) -> str:
    """text casefolded with the Turkish dotted and dotless i folded together (ı -> i, and the combining dot U+0307 that
    casefold leaves on İ dropped, so a case change cannot hide a copy), punctuation (Unicode category P*) removed and
    whitespace runs collapsed to one space."""
    folded = text.casefold().replace("\u0307", "").replace("ı", "i")
    kept = "".join(ch for ch in folded if not unicodedata.category(ch).startswith("P"))
    return " ".join(kept.split())


@pytest.mark.data
def test_examples_are_not_dataset_text() -> None:
    examples = example_strings()
    assert len(examples) == 6, "ui/src/examples.ts holds %d double-quoted strings, not 6" % len(examples)
    split_dir = os.path.join(REPO, "data", "splits")
    names = sorted(n for n in os.listdir(split_dir) if n.endswith(".jsonl")) if os.path.isdir(split_dir) else []
    if not names:
        pytest.skip("data/splits holds no *.jsonl to compare the examples against; run `make data`")
    wanted = {normalize(e): i for i, e in enumerate(examples)}
    hits = []
    for name in names:
        with open(os.path.join(split_dir, name), "rb") as f:
            rows = pilot.data.parse_jsonl(f.read())
        assert rows, "%s holds no row" % name
        found = {wanted[k] for k in {normalize(r["state"]) for r in rows} if k in wanted}
        # only the example's index and the split file name: never a split text
        hits += ["example %d in %s" % (i, name) for i in sorted(found)]
    assert hits == [], "examples equal to a split message: %s" % hits


def test_examples_cover_turkish_letters() -> None:
    examples = example_strings()
    assert len(examples) == 6
    joined = "".join(examples)
    missing = [ch for ch in TURKISH_LETTERS if ch not in joined]
    assert missing == [], "the examples together lack %s" % missing


def test_chat_copy_contract() -> None:
    texts = []
    for rel in ui_sources():
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            texts.append(f.read())
    source = "\n".join(texts)
    missing = [c for c in CHAT_COPY if c not in source]
    assert missing == [], "chat copy missing from ui/src: %s" % missing


# --- the dashboard: copy and routes ------------------------------------------------------------------------------

# 11-UI-SPEC.md Copywriting Contract and "Dashboard view": the dashboard and header copy, each exactly as written there
DASHBOARD_COPY = (
    "Training runs",
    "Val selection",
    "Test results and verdict",
    "Confidence, latency and few-shot",
    "View the frozen render",
    "Superseded, never current",
    "Show superseded runs",
    "Training curves unavailable",
    "Every other dashboard view reads results/ and still works.",
    "Reported, not gated",
    "Val (in-sample)",
    "Diagram unavailable",
    "Verdict unavailable",
    "Could not read",
    "Loss values are not comparable across the two models: they train on different objectives.",
    "Local demo · stores nothing",
    "Laya vs BERTurk: Turkish banking intent demo",
)
# (route key, sidebar label) of the four dashboard views, in the UI-SPEC's sidebar order
DASHBOARD_VIEWS = (("training", "Training runs"), ("selection", "Val selection"),
                   ("verdict", "Test results and verdict"), ("confidence", "Confidence, latency and few-shot"))
VIEW_ENTRY_RE = re.compile(r'\{\s*key:\s*"([a-z]+)",\s*label:\s*"([^"]+)"\s*\}')
VIEW_COMPONENTS_RE = re.compile(r"export const VIEW_COMPONENTS[^=]*=\s*\{([^}]*)\}", re.S)
COMPONENT_ENTRY_RE = re.compile(r"^\s*([a-z]+):\s*([A-Za-z]+),?\s*$", re.M)


def ui_source_text() -> str:
    """Every file of ui_sources() joined by newlines."""
    texts = []
    for rel in ui_sources():
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            texts.append(f.read())
    return "\n".join(texts)


def test_dashboard_copy_contract() -> None:
    source = ui_source_text()
    missing = [c for c in DASHBOARD_COPY if c not in source]
    assert missing == [], "dashboard copy missing from ui/src: %s" % missing


def test_dashboard_routes() -> None:
    with open(os.path.join(UI, "src", "App.tsx"), encoding="utf-8") as f:
        app = f.read()
    assert '"#/chat"' in app, "ui/src/App.tsx does not route #/chat"
    with open(os.path.join(UI, "src", "dashboard", "Dashboard.tsx"), encoding="utf-8") as f:
        dash = f.read()
    assert '"#/dashboard/"' in dash, "Dashboard.tsx does not build the #/dashboard/ view hashes"
    assert tuple(VIEW_ENTRY_RE.findall(dash)) == DASHBOARD_VIEWS, "Dashboard.tsx VIEWS differ from the UI-SPEC"
    table = VIEW_COMPONENTS_RE.search(dash)
    assert table is not None, "Dashboard.tsx has no VIEW_COMPONENTS table"
    components = dict(COMPONENT_ENTRY_RE.findall(table.group(1)))
    missing = [k for k, _ in DASHBOARD_VIEWS if k not in components]
    assert missing == [], "views without a component in VIEW_COMPONENTS: %s" % missing
    for key, component in components.items():
        assert os.path.isfile(os.path.join(UI, "src", "dashboard", component + ".tsx")), \
            "VIEW_COMPONENTS maps %s to %s, which has no ui/src/dashboard/%s.tsx" % (key, component, component)
        assert re.search(r'^import %s from "\./%s";$' % (component, component), dash, re.M), \
            "Dashboard.tsx does not import %s" % component


# --- the status polls: a failed poll is retried, never terminal (11-REVIEW WR-01) --------------------------------

def read_ui(rel: str) -> str:
    """The text of ui/src/<rel>."""
    with open(os.path.join(UI, "src", rel), encoding="utf-8") as f:
        return f.read()


def hook_body(rel: str, name: str) -> str:
    """The text of `function name(` (or `export function name<`) in ui/src/<rel> up to the next line holding only "}"
    (the hook's closing brace)."""
    m = re.search(r"^(?:export )?function %s[(<].*?^\}$" % name, read_ui(rel), re.S | re.M)
    assert m is not None, "ui/src/%s has no function %s" % (rel, name)
    return m.group(0)


def test_status_polls_retry_after_a_failure() -> None:
    """useChatMode reschedules its poll on a failed /api/status answer (unreachable or an HTTP error) with a capped
    backoff and keeps polling while the chat reads refused (11-REVIEW WR-07: a server refusal is fixed by a restart of
    `make demo`, which the page must pick up), a failed classify re-reads /api/status (so a chat the server refused
    shows as refused), useTraining reschedules its status poll on an unanswered request and on an HTTP error, useView
    and useIntents fetch again until they answer: none ends in a state only a reload leaves, so no copy tells the user
    to reload."""
    chat = hook_body("chat/ChatView.tsx", "useChatMode")
    failed = re.search(r"if \(!r\.ok\) \{(.*?)\n      \}", chat, re.S)
    assert failed is not None, "useChatMode has no failed-poll branch"
    assert "window.setTimeout(() => void poll(), retry)" in failed.group(1), "a failed status poll is not retried"
    assert "RETRY_MAX_MS" in failed.group(1), "the status retry has no backoff cap"
    refused = re.search(r'\} else if \(next\.kind === "refused"\) \{(.*?)\n      \}', chat, re.S)
    assert refused is not None and "window.setTimeout(() => void poll(), RETRY_MAX_MS)" in refused.group(1), \
        "a refused chat never polls /api/status again (11-REVIEW WR-07)"
    view = hook_body("dashboard/common.tsx", "useView")
    assert "window.setTimeout(() => void fetchView(), retry)" in view, "an unanswered view request is not retried"
    assert "window.setTimeout(() => void fetchView(), RETRY_MAX_MS)" in view, "a refused view is never asked again"
    intents = hook_body("chat/ChatView.tsx", "useIntents")
    assert "window.setTimeout(() => void fetchIntents(), RETRY_MAX_MS)" in intents, "/api/intents is not retried"
    send = re.search(r"const send = useCallback\((.*?)\n  \);", read_ui("chat/ChatView.tsx"), re.S)
    assert send is not None and re.search(r"if \(!r\.ok\) \{\s*recheck\(\);", send.group(1)), \
        "a failed classify does not re-read /api/status (11-REVIEW WR-05)"
    training = hook_body("dashboard/TrainingView.tsx", "useTraining")
    unanswered = re.search(r"if \(r\.status === null\) \{(.*?)\n      \}", training, re.S)
    assert unanswered is not None, "useTraining has no unanswered-request branch"
    assert "window.setTimeout(() => void pollStatus(), retry)" in unanswered.group(1), \
        "an unanswered training request is not retried"
    assert "window.setTimeout(() => void pollStatus(), RETRY_MAX_MS)" in training, \
        "a training request answered with an HTTP error is never asked again"
    said = [rel for rel in ui_sources() if "reload this page" in read_ui(os.path.relpath(rel, "ui/src"))]
    assert said == [], "UI copy still tells the user to reload, though every state recovers: %s" % said


# --- the per-view error: the server's line verbatim, no fix of the UI's (11-REVIEW WR-03) -------------------------

def test_view_error_adds_no_fix_of_its_own() -> None:
    """The server's one line ends in its own fix, and VIEW_SOURCES names only a view's primary record, so no ui/src
    file adds a `git checkout` command, and no error state invents a source from the endpoint path."""
    hits = []
    for rel in ui_sources():
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            text = f.read()
        hits += ["%s: git checkout" % rel for _ in re.findall(r"git checkout", text)]
        hits += ["%s: %s" % (rel, m) for m in re.findall(r'source: r\.source \?\? (?!null\b)[^ ;}]+', text)]
    assert hits == [], "the UI adds a fix or a source of its own: %s" % hits
