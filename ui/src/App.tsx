// The app shell: the header and the live chat status read from GET /api/status (demo.py). While the six final seeds
// load the status is polled every second; a startup refusal shows the server's one line verbatim; a server that does
// not answer shows the unreachable copy. The chat thread of the ready state is rendered by a later plan (11-06).
// Every string the server sends is rendered as a React text node.

import { useEffect, useState, type CSSProperties } from "react";
import { getJSON } from "./http";

type ChatState = "loading" | "ready" | "refused";

interface Status {
  chat: ChatState;
  message: string | null;
  device: string;
}

type View =
  | { kind: "loading" }
  | { kind: "ready"; device: string }
  | { kind: "refused"; message: string }
  | { kind: "unreachable" };

const POLL_MS = 1000;

const TITLE = "Laya vs BERTurk: Turkish banking intent demo";
const NOTE = "Local demo · stores nothing";
const LOADING_COPY =
  "Loading the six final checkpoints… The chat opens when they are ready; the dashboard already works.";
const UNAVAILABLE_HEADING = "Chat unavailable";
const UNAVAILABLE_TAIL = "The dashboard still works.";
const UNREACHABLE_COPY =
  "The demo server is not responding. Check that `make demo` is still running in your terminal, then reload this page.";

const styles: Record<string, CSSProperties> = {
  header: {
    height: "var(--header-height)",
    background: "var(--color-surface)",
    borderBottom: "1px solid var(--color-hairline)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "var(--space-md)",
    padding: "0 var(--space-lg)",
  },
  title: {
    margin: 0,
    fontSize: "var(--text-heading)",
    fontWeight: "var(--weight-semibold)",
    lineHeight: "var(--leading-heading)",
  },
  note: {
    fontSize: "var(--text-label)",
    lineHeight: "var(--leading-label)",
    color: "var(--ink-muted)",
  },
  main: {
    maxWidth: "var(--chat-max)",
    margin: "0 auto",
    padding: "var(--space-3xl) var(--space-lg) var(--space-lg)",
  },
  card: {
    background: "var(--color-surface)",
    border: "1px solid var(--color-hairline)",
    borderRadius: "var(--radius-card)",
    padding: "var(--space-md)",
  },
  body: {
    margin: 0,
    fontSize: "var(--text-body)",
    lineHeight: "var(--leading-body)",
  },
  heading: {
    margin: "0 0 var(--space-sm)",
    fontSize: "var(--text-heading)",
    fontWeight: "var(--weight-semibold)",
    lineHeight: "var(--leading-heading)",
  },
  refusal: {
    margin: "0 0 var(--space-sm)",
    padding: "var(--space-sm) var(--space-md)",
    background: "var(--color-code)",
    borderRadius: "var(--radius-card)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-label)",
    lineHeight: "var(--leading-label)",
    whiteSpace: "pre-wrap",
    overflowWrap: "anywhere",
  },
  secondary: {
    margin: 0,
    fontSize: "var(--text-body)",
    lineHeight: "var(--leading-body)",
    color: "var(--ink-secondary)",
  },
};

function toView(s: Status): View {
  if (s.chat === "ready") {
    return { kind: "ready", device: s.device };
  }
  if (s.chat === "refused") {
    return { kind: "refused", message: s.message ?? "" };
  }
  return { kind: "loading" };
}

function useChatStatus(): View {
  const [view, setView] = useState<View>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async (): Promise<void> => {
      const r = await getJSON<Status>("/api/status");
      if (cancelled) {
        return;
      }
      if (!r.ok) {
        setView({ kind: "unreachable" });
        return;
      }
      const next = toView(r.data);
      setView(next);
      if (next.kind === "loading") {
        timer = window.setTimeout(poll, POLL_MS);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, []);
  return view;
}

function ChatStatus({ view }: { view: View }) {
  if (view.kind === "loading") {
    return (
      <div style={styles.card} role="status" aria-live="polite">
        <p style={styles.body}>{LOADING_COPY}</p>
      </div>
    );
  }
  if (view.kind === "refused") {
    return (
      <div style={styles.card} role="alert">
        <h2 style={styles.heading}>{UNAVAILABLE_HEADING}</h2>
        <p style={styles.refusal}>{view.message}</p>
        <p style={styles.secondary}>{UNAVAILABLE_TAIL}</p>
      </div>
    );
  }
  if (view.kind === "unreachable") {
    return (
      <div style={styles.card} role="alert">
        <p style={styles.body}>{UNREACHABLE_COPY}</p>
      </div>
    );
  }
  return null;
}

export default function App() {
  const view = useChatStatus();
  return (
    <>
      <header style={styles.header}>
        <h1 style={styles.title}>{TITLE}</h1>
        <span style={styles.note}>{NOTE}</span>
      </header>
      <main style={styles.main}>
        <ChatStatus view={view} />
      </main>
    </>
  );
}
