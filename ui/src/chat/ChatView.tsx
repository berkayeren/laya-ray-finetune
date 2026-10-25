// The chat view (D-07, D-13): a thread of turns, each the user's message and one reply card holding both systems'
// answers, and the sticky input bar (InputBar). A message is posted unchanged as {text} to POST /api/classify (UTF-8
// JSON); one request is in flight at a time. The thread lives in React state only: nothing is written to any browser
// store or cookie, no URL holds message text, and a reload empties the thread. GET /api/intents is read once: its
// INTENTS tag order (the order of every answer's arrays) and each tag's question text, passed down as one tag lookup
// (the chips' and the 48-tag rows' tooltips), and its confidence_rule, which every reply card shows on the confidence
// info icon. While the response is missing these are omitted, never invented. Every server string is rendered as a
// React text node.
//
// ChatView owns the chat area's states. It polls GET /api/status every second while the status reads loading:
//   loading      the empty state and the examples stay visible, the input is disabled and says the six final
//                checkpoints are loading (the dashboard already works);
//   refused      "Chat unavailable", the server's one line verbatim in mono, and a link to the dashboard;
//   unreachable  the server-unreachable copy;
//   ready        the chat.
// A pending turn shows the loading reply card (two column skeletons); a failed request keeps the user bubble and shows
// the request-error copy (or the unreachable copy when the server did not answer) in place of the reply card. The
// backend answers with both systems or neither, so a reply card never shows one column alone. The thread is a
// role="log"; the view follows the newest turn only when the user was already at the bottom.
// ChatView takes no props (plan 11-10 relies on that signature).

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { getJSON, postJSON } from "../http";
import InputBar from "./InputBar";
import ReplyCard from "./ReplyCard";
import type { TagLookup } from "./TagPanel";
import type { ClassifyResponse, Intents, Status } from "./types";
import css from "./chat.module.css";

const POLL_MS = 1000;
const BOTTOM_SLACK_PX = 24;
const EMPTY_HEADING = "Classify a Turkish banking message";
const EMPTY_BODY =
  "Type a message in Turkish or pick an example below. Fine-tuned Laya and fine-tuned BERTurk each answer side by " +
  "side, with their predicted intent tags, probabilities and message confidence.";
const PENDING_COPY = "Laya and BERTurk are classifying…";
const REPLY_RECEIVED = "Reply received";
const LOADING_COPY =
  "Loading the six final checkpoints… The chat opens when they are ready; the dashboard already works.";
const UNAVAILABLE_HEADING = "Chat unavailable";
const UNAVAILABLE_TAIL = "The dashboard still works.";
const DASHBOARD_LINK = "Open the dashboard";
const DASHBOARD_HREF = "#/dashboard/verdict";
const UNREACHABLE_COPY =
  "The demo server is not responding. Check that `make demo` is still running in your terminal, then reload this page.";

function requestError(message: string): string {
  return `Classification failed: ${message}. Try again; if it repeats, restart \`make demo\`.`;
}

type ChatMode =
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "refused"; message: string }
  | { kind: "unreachable" };

// error: the server's one-line message, or null when the server did not answer
type Turn =
  | { id: number; text: string; state: "pending" }
  | { id: number; text: string; state: "done"; reply: ClassifyResponse }
  | { id: number; text: string; state: "error"; error: string | null };

function toMode(s: Status): ChatMode {
  if (s.chat === "ready") {
    return { kind: "ready" };
  }
  if (s.chat === "refused") {
    return { kind: "refused", message: s.message ?? "" };
  }
  return { kind: "loading" };
}

// GET /api/status, polled every second while it reads loading; an unanswered poll ends in unreachable
function useChatMode(): ChatMode {
  const [mode, setMode] = useState<ChatMode>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async (): Promise<void> => {
      const r = await getJSON<Status>("/api/status");
      if (cancelled) {
        return;
      }
      if (!r.ok) {
        setMode({ kind: "unreachable" });
        return;
      }
      const next = toMode(r.data);
      setMode(next);
      if (next.kind === "loading") {
        timer = window.setTimeout(() => void poll(), POLL_MS);
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
  return mode;
}

function useIntents(): Intents | null {
  const [intents, setIntents] = useState<Intents | null>(null);
  useEffect(() => {
    let cancelled = false;
    void getJSON<Intents>("/api/intents").then((r) => {
      if (!cancelled && r.ok) {
        setIntents(r.data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return intents;
}

// true when the page is scrolled to within BOTTOM_SLACK_PX of its bottom
function atBottom(): boolean {
  const el = document.documentElement;
  return window.innerHeight + window.scrollY >= el.scrollHeight - BOTTOM_SLACK_PX;
}

// the loading reply card: two column skeletons (a title row and three 28px grey bars) and the pending copy
function LoadingCard() {
  return (
    <div className={css.card}>
      <p className={css.pendingLine}>{PENDING_COPY}</p>
      {(["Laya", "BERTurk"] as const).map((name) => (
        <div key={name} className={css.column} aria-hidden="true">
          <div className={css.skeletonTitle} />
          <div className={css.skeletonBar} />
          <div className={css.skeletonBar} />
          <div className={css.skeletonBar} />
        </div>
      ))}
    </div>
  );
}

interface TurnProps {
  turn: Turn;
  tags: TagLookup | null;
  confidence_rule: string | null;
}

function TurnView({ turn, tags, confidence_rule }: TurnProps) {
  return (
    <li className={css.turn}>
      <p className={css.bubble} lang="tr">
        {turn.text}
      </p>
      <div className={css.replySlot}>
        {turn.state === "pending" && <LoadingCard />}
        {turn.state === "done" && <ReplyCard reply={turn.reply} tags={tags} confidence_rule={confidence_rule} />}
        {turn.state === "error" && (
          <p className={css.error} role="alert">
            {turn.error === null ? UNREACHABLE_COPY : requestError(turn.error)}
          </p>
        )}
        <span className={css.srOnly} aria-live="polite">
          {turn.state === "done" ? REPLY_RECEIVED : ""}
        </span>
      </div>
    </li>
  );
}

function Unavailable({ message }: { message: string }) {
  return (
    <div className={css.notice} role="alert">
      <h2 className={css.noticeHeading}>{UNAVAILABLE_HEADING}</h2>
      <p className={css.refusal}>{message}</p>
      <p className={css.noticeBody}>
        {UNAVAILABLE_TAIL} <a className={css.link} href={DASHBOARD_HREF}>{DASHBOARD_LINK}</a>
      </p>
    </div>
  );
}

export default function ChatView() {
  const mode = useChatMode();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const nextId = useRef(0);
  const follow = useRef(true);
  const intents = useIntents();
  const tags = useMemo<TagLookup | null>(
    () =>
      intents === null
        ? null
        : new Map(intents.tags.map((t, index) => [t.tag, { index, question: t.question }] as const)),
    [intents],
  );
  const confidence_rule = intents === null ? null : intents.confidence_rule;
  const ready = mode.kind === "ready";

  // whether the user is at the bottom, read on every scroll (content growth fires no scroll event)
  useEffect(() => {
    const onScroll = (): void => {
      follow.current = atBottom();
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // follow the newest turn only when the user was already at the bottom
  useLayoutEffect(() => {
    if (follow.current && turns.length > 0) {
      window.scrollTo({ top: document.documentElement.scrollHeight });
    }
  }, [turns]);

  const send = useCallback(
    async (message: string): Promise<void> => {
      if (!ready || inFlight.current || message.trim() === "") {
        return;
      }
      inFlight.current = true;
      setBusy(true);
      const id = nextId.current++;
      setTurns((ts) => [...ts, { id, text: message, state: "pending" }]);
      setText("");
      const r = await postJSON<ClassifyResponse>("/api/classify", { text: message });
      inFlight.current = false;
      setBusy(false);
      // a turn removed by "Clear conversation" while its request was in flight stays removed
      setTurns((ts) =>
        ts.map((t): Turn => {
          if (t.id !== id) {
            return t;
          }
          if (r.ok) {
            return { id, text: message, state: "done", reply: r.data };
          }
          return { id, text: message, state: "error", error: r.status === null ? null : r.error };
        }),
      );
    },
    [ready],
  );

  const clear = useCallback((): void => {
    setTurns([]);
    setText("");
  }, []);

  return (
    <div className={css.view}>
      {mode.kind === "refused" && <Unavailable message={mode.message} />}
      {mode.kind === "unreachable" && (
        <div className={css.notice} role="alert">
          <p className={css.noticeBody}>{UNREACHABLE_COPY}</p>
        </div>
      )}
      {turns.length === 0 && (mode.kind === "loading" || mode.kind === "ready") && (
        <div className={css.empty}>
          <h2 className={css.emptyHeading}>{EMPTY_HEADING}</h2>
          <p className={css.emptyBody}>{EMPTY_BODY}</p>
        </div>
      )}
      <ol className={css.thread} role="log" aria-label="Conversation">
        {turns.map((t) => (
          <TurnView key={t.id} turn={t} tags={tags} confidence_rule={confidence_rule} />
        ))}
      </ol>
      <InputBar
        text={text}
        onText={setText}
        onSend={(m) => void send(m)}
        ready={ready}
        busy={busy}
        status={mode.kind === "loading" ? LOADING_COPY : null}
        hasTurns={turns.length > 0}
        onClear={clear}
      />
    </div>
  );
}
