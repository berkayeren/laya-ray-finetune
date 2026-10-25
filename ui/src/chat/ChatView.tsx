// The chat view (D-07, D-13): a thread of turns, each the user's message and one reply card holding both systems'
// answers, and the input. A message is posted unchanged as {text} to POST /api/classify; one request is in flight at a
// time. The thread lives in React state only: nothing is written to any browser store or cookie, no URL holds message
// text, and a reload empties the thread. GET /api/intents is read once for the INTENTS tag order (the order of every
// answer's probabilities). Every server string is rendered as a React text node.
// ChatView takes no props (plans 11-09 and 11-10 rely on that signature).

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { getJSON, postJSON } from "../http";
import ReplyCard from "./ReplyCard";
import type { ClassifyResponse, Intents } from "./types";
import css from "./chat.module.css";

const MAX_CHARS = 500;
const CTA = "Classify message";
const PLACEHOLDER = "Type a Turkish banking message";
const EMPTY_HEADING = "Classify a Turkish banking message";
const EMPTY_BODY =
  "Type a message in Turkish or pick an example below. Fine-tuned Laya and fine-tuned BERTurk each answer side by " +
  "side, with their predicted intent tags, probabilities and message confidence.";
const CONTEXT_NOTE =
  "Each message is classified on its own: the models see no earlier messages. This local demo stores and logs nothing.";
const PENDING_COPY = "Laya and BERTurk are classifying…";
const UNREACHABLE_COPY =
  "The demo server is not responding. Check that `make demo` is still running in your terminal, then reload this page.";

function requestError(message: string): string {
  return `Classification failed: ${message}. Try again; if it repeats, restart \`make demo\`.`;
}

// error: the server's one-line message, or null when the server did not answer
type Turn =
  | { id: number; text: string; state: "pending" }
  | { id: number; text: string; state: "done"; reply: ClassifyResponse }
  | { id: number; text: string; state: "error"; error: string | null };

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

function TurnView({ turn, tagIndex }: { turn: Turn; tagIndex: ReadonlyMap<string, number> | null }) {
  return (
    <li className={css.turn}>
      <p className={css.bubble} lang="tr">
        {turn.text}
      </p>
      {turn.state === "pending" && (
        <p className={css.pending} role="status">
          {PENDING_COPY}
        </p>
      )}
      {turn.state === "done" && <ReplyCard reply={turn.reply} tagIndex={tagIndex} />}
      {turn.state === "error" && (
        <p className={css.error} role="alert">
          {turn.error === null ? UNREACHABLE_COPY : requestError(turn.error)}
        </p>
      )}
    </li>
  );
}

export default function ChatView() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const nextId = useRef(0);
  const intents = useIntents();
  const tagIndex = useMemo(
    () => (intents === null ? null : new Map(intents.tags.map((t, i) => [t.tag, i] as const))),
    [intents],
  );
  const busy = turns.some((t) => t.state === "pending");
  const disabled = busy || text.trim() === "";

  const send = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (disabled) {
      return;
    }
    const id = nextId.current++;
    const message = text;
    setTurns((ts) => [...ts, { id, text: message, state: "pending" }]);
    setText("");
    const r = await postJSON<ClassifyResponse>("/api/classify", { text: message });
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
  };

  return (
    <div className={css.view}>
      {turns.length === 0 ? (
        <div className={css.empty}>
          <h2 className={css.emptyHeading}>{EMPTY_HEADING}</h2>
          <p className={css.emptyBody}>{EMPTY_BODY}</p>
        </div>
      ) : (
        <ol className={css.thread}>
          {turns.map((t) => (
            <TurnView key={t.id} turn={t} tagIndex={tagIndex} />
          ))}
        </ol>
      )}
      <form className={css.inputBar} onSubmit={(e) => void send(e)}>
        <div className={css.inputRow}>
          <textarea
            className={css.textarea}
            lang="tr"
            rows={1}
            maxLength={MAX_CHARS}
            placeholder={PLACEHOLDER}
            aria-label="Message"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button className={css.cta} type="submit" disabled={disabled}>
            {CTA}
          </button>
        </div>
        <p className={css.note}>{CONTEXT_NOTE}</p>
      </form>
    </div>
  );
}
