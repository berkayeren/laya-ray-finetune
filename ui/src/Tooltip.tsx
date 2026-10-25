// A tooltip on hover and on keyboard focus (11-UI-SPEC.md Interaction Contract, "Tooltips"): it opens 200 ms after
// the pointer enters or focus arrives, and closes on pointer leave, blur or Escape. It is at most 320px wide, has
// role="tooltip", and the trigger (the one child element, which must be focusable) gets aria-describedby pointing at
// it, so a screen reader reads the text on focus. The tooltip element stays in the DOM (hidden while closed) so the
// description is always resolvable. The content is a plain string rendered as a text node: never markup.

import {
  cloneElement,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from "react";

const DELAY_MS = 200;

const styles: Record<string, CSSProperties> = {
  anchor: {
    position: "relative",
    display: "inline-flex",
  },
  tip: {
    position: "absolute",
    bottom: "calc(100% + var(--space-sm))",
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 10,
    width: "max-content",
    maxWidth: "320px",
    padding: "var(--space-sm) var(--space-md)",
    background: "var(--ink-primary)",
    color: "var(--color-surface)",
    borderRadius: "var(--radius-card)",
    fontSize: "var(--text-label)",
    fontWeight: "var(--weight-regular)",
    lineHeight: "var(--leading-label)",
    whiteSpace: "normal",
    overflowWrap: "anywhere",
    pointerEvents: "none",
  },
};

interface TriggerProps {
  "aria-describedby"?: string;
}

interface TooltipProps {
  text: string;
  children: ReactElement<TriggerProps>;
}

export default function Tooltip({ text, children }: TooltipProps) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  const cancel = useCallback((): void => {
    if (timer.current !== undefined) {
      window.clearTimeout(timer.current);
      timer.current = undefined;
    }
  }, []);
  const show = useCallback((): void => {
    cancel();
    timer.current = window.setTimeout(() => setOpen(true), DELAY_MS);
  }, [cancel]);
  const hide = useCallback((): void => {
    cancel();
    setOpen(false);
  }, [cancel]);

  // Escape closes the tooltip wherever focus is (a hover-opened tooltip has no focused trigger)
  useEffect(() => {
    if (!open) {
      return;
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") {
        hide();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, hide]);

  useEffect(() => cancel, [cancel]);

  return (
    <span style={styles.anchor} onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {cloneElement(children, { "aria-describedby": id })}
      <span id={id} role="tooltip" hidden={!open} style={styles.tip}>
        {text}
      </span>
    </span>
  );
}
