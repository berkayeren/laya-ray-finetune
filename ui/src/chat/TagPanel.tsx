// The ranked 48-tag panel of one reply column (D-04, D-08): one 28px row per tag, in the answer's ranked order
// (probability descending, ties in INTENTS order, as the server sends it). A row holds the tag name (600 weight with a
// check icon when the tag is in the answer's predicted list, 400 otherwise), an 8px track 0..1 whose fill is the
// probability in the column's series colour, a 2px tick at that tag's θ and the probability. Its caption, shown on
// hover and focus, names the θ and whether it is the tag's own θ or the global one, from the answer's own_theta flags.
// The tooltip of a row is the tag's question text exactly as GET /api/intents serves it (English, never translated).
// The UI never sorts and never compares a probability with θ: the order and the predicted flags come from the answer.
// A tag's probability, θ and own_theta flag are read at its index in the INTENTS order of GET /api/intents (the order
// of the answer's arrays); while that response is missing a row shows its tag name only and no tooltip.

import type { CSSProperties, ReactElement } from "react";
import { formatProbability, formatTheta } from "../format";
import { CheckIcon } from "../icons";
import Tooltip from "../Tooltip";
import type { Answer } from "./types";
import css from "./chat.module.css";

// one INTENTS tag: its index in the answer's arrays and its question text as stored
export interface TagInfo {
  index: number;
  question: string;
}

export type TagLookup = ReadonlyMap<string, TagInfo>;

// The caption of a row: the θ as recorded and its source.
export function thetaCaption(theta: number, own: boolean): string {
  return own ? `θ ${formatTheta(theta)} (own θ)` : `θ ${formatTheta(theta)} (global θ)`;
}

// a focusable element the Tooltip can describe
type Trigger = ReactElement<{ "aria-describedby"?: string }>;

// The question tooltip around one focusable element, or the element alone while the question is missing.
export function withQuestion(tags: TagLookup | null, tag: string, child: Trigger): ReactElement {
  const info = tags?.get(tag);
  return info === undefined ? child : <Tooltip text={info.question}>{child}</Tooltip>;
}

interface RowProps {
  tag: string;
  predicted: boolean;
  answer: Answer;
  tags: TagLookup | null;
}

function Row({ tag, predicted, answer, tags }: RowProps) {
  const info = tags?.get(tag);
  const name = (
    <span className={predicted ? css.tagNamePredicted : css.tagName}>
      {predicted && <CheckIcon />}
      <span className={css.tagText}>{tag}</span>
    </span>
  );
  if (info === undefined) {
    return (
      <div className={css.tagRow} tabIndex={0}>
        {name}
      </div>
    );
  }
  const p = answer.probabilities[info.index];
  const theta = answer.theta[info.index];
  return (
    <Tooltip text={info.question}>
      <div className={css.tagRow} tabIndex={0}>
        {name}
        <span className={css.tagTrack} aria-hidden="true">
          <span className={css.tagFill} style={{ width: `${p * 100}%` }} />
          <span className={css.tagTick} style={{ left: `${theta * 100}%` }} />
        </span>
        <span className={css.tagValue}>{formatProbability(p)}</span>
        <span className={css.tagCaption}>{thetaCaption(theta, answer.own_theta[info.index])}</span>
      </div>
    </Tooltip>
  );
}

interface TagPanelProps {
  id: string;
  answer: Answer;
  series: string;
  tags: TagLookup | null;
  open: boolean;
}

export default function TagPanel({ id, answer, series, tags, open }: TagPanelProps) {
  const style = { "--series": series } as CSSProperties;
  const predicted = new Set(answer.predicted);
  return (
    <ol id={id} className={css.tagPanel} style={style} hidden={!open}>
      {open &&
        answer.ranked.map((tag) => (
          <li key={tag} className={css.tagItem}>
            <Row tag={tag} predicted={predicted.has(tag)} answer={answer} tags={tags} />
          </li>
        ))}
    </ol>
  );
}
