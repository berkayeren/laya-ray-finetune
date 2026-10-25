// The sticky input bar of the chat view (11-UI-SPEC.md "Chat view" input bar, Interaction Contract, Copywriting):
//   row 1: "Try an example" and one chip per demo-authored example (ui/src/examples.ts); a click fills the textarea
//          and sends that text at once;
//   row 2: the textarea (lang="tr", 44px growing to 160px, then it scrolls; at most 500 characters) and the
//          "Classify message" button. Enter sends, Shift+Enter inserts a newline, and no key sends while an IME
//          composition is open. From 400 characters a counter "n / 500" shows; at 500 it names the limit;
//   row 3: the context note and, once the thread has a turn, "Clear conversation", which turns into an inline confirm
//          ("Clear messages" / "Keep messages") for 6 seconds.
// Sending is possible only while the chat is ready and no request is in flight (one at a time), and never for an empty
// or whitespace-only text. The text lives in the parent's React state only (D-13): nothing here stores it.

import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { EXAMPLES } from "../examples";
import { SendIcon } from "../icons";
import css from "./chat.module.css";

const MAX_CHARS = 500;
const COUNTER_FROM = 400;
const TEXTAREA_MAX_PX = 160;
const CONFIRM_MS = 6000;
const CTA = "Classify message";
const PLACEHOLDER = "Type a Turkish banking message";
const EXAMPLES_LABEL = "Try an example";
const CONTEXT_NOTE =
  "Each message is classified on its own: the models see no earlier messages. This local demo stores and logs nothing.";
const LIMIT_COPY = "500 / 500: the limit for one message";
const CLEAR = "Clear conversation";
const CLEAR_CONFIRM = "Clear messages";
const CLEAR_KEEP = "Keep messages";
const CLEAR_CAPTION = "Remove every message from this tab? Nothing is stored, so they cannot be restored.";

interface InputBarProps {
  text: string;
  onText: (text: string) => void;
  onSend: (text: string) => void;
  ready: boolean;
  busy: boolean;
  status: string | null;
  hasTurns: boolean;
  onClear: () => void;
}

// The counter text for a text of n characters, or null below the threshold.
function counter(n: number): string | null {
  if (n >= MAX_CHARS) {
    return LIMIT_COPY;
  }
  return n >= COUNTER_FROM ? `${n} / ${MAX_CHARS}` : null;
}

function ClearConversation({ onClear }: { onClear: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const keep = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (!confirming) {
      return;
    }
    keep.current?.focus();
    const timer = window.setTimeout(() => setConfirming(false), CONFIRM_MS);
    return () => window.clearTimeout(timer);
  }, [confirming]);
  if (!confirming) {
    return (
      <button type="button" className={css.textButton} onClick={() => setConfirming(true)}>
        {CLEAR}
      </button>
    );
  }
  return (
    <span className={css.confirm} role="group" aria-label={CLEAR}>
      <span className={css.confirmCaption}>{CLEAR_CAPTION}</span>
      <button type="button" className={css.destructive} onClick={() => { setConfirming(false); onClear(); }}>
        {CLEAR_CONFIRM}
      </button>
      <button type="button" ref={keep} className={css.textButton} onClick={() => setConfirming(false)}>
        {CLEAR_KEEP}
      </button>
    </span>
  );
}

export default function InputBar({ text, onText, onSend, ready, busy, status, hasTurns, onClear }: InputBarProps) {
  const area = useRef<HTMLTextAreaElement | null>(null);
  const canSend = ready && !busy;
  const sendable = canSend && text.trim() !== "";
  const count = counter(text.length);

  // grow from the 44px minimum with the text, up to 160px, then scroll
  useLayoutEffect(() => {
    const el = area.current;
    if (el === null) {
      return;
    }
    el.style.height = "auto";
    const full = el.scrollHeight + el.offsetHeight - el.clientHeight;
    el.style.height = `${Math.min(full, TEXTAREA_MAX_PX)}px`;
    el.style.overflowY = full > TEXTAREA_MAX_PX ? "auto" : "hidden";
  }, [text]);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    // an open IME composition (isComposing, or Safari's keyCode 229) never sends; Shift+Enter inserts a newline
    if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing || e.keyCode === 229) {
      return;
    }
    e.preventDefault();
    if (sendable) {
      onSend(text);
    }
  };

  const pick = (example: string): void => {
    if (!canSend) {
      return;
    }
    onText(example);
    onSend(example);
  };

  return (
    <form
      className={css.inputBar}
      onSubmit={(e) => {
        e.preventDefault();
        if (sendable) {
          onSend(text);
        }
      }}
    >
      <div className={css.examples}>
        <span className={css.examplesLabel}>{EXAMPLES_LABEL}</span>
        <ul className={css.exampleList}>
          {EXAMPLES.map((example) => (
            <li key={example}>
              <button type="button" className={css.example} lang="tr" disabled={!canSend} onClick={() => pick(example)}>
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>
      {status !== null && (
        <p className={css.status} role="status">
          {status}
        </p>
      )}
      <div className={css.inputRow}>
        <textarea
          ref={area}
          className={css.textarea}
          lang="tr"
          rows={1}
          maxLength={MAX_CHARS}
          placeholder={PLACEHOLDER}
          aria-label="Message"
          disabled={!ready}
          value={text}
          onChange={(e) => onText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button className={css.cta} type="submit" disabled={!sendable}>
          <SendIcon />
          {CTA}
        </button>
      </div>
      {count !== null && (
        <p className={text.length >= MAX_CHARS ? css.counterLimit : css.counter} aria-live="polite">
          {count}
        </p>
      )}
      <div className={css.noteRow}>
        <p className={css.note}>{CONTEXT_NOTE}</p>
        {hasTurns && (
          <ClearConversation
            onClear={() => {
              onClear();
              area.current?.focus();
            }}
          />
        )}
      </div>
    </form>
  );
}
