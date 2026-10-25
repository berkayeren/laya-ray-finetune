// One reply card (D-04, D-05): Laya left, BERTurk right, always in that order. Each column shows the system's settings
// line, its predicted tags as chips (in the order of the answer's predicted list) or the No intent block, its message
// confidence with an info icon whose tooltip is the frozen confidence rule, exactly as GET /api/intents serves it
// (confidence_rule; while it is missing the icon is omitted, and the UI writes no rule text of its own), and the
// "Show all 48 tags" disclosure (TagPanel). The card holds one open state, so both columns' disclosures open and close
// together. Every number comes from the answer: the UI computes no threshold, average or confidence, and it never
// sorts or compares a probability with θ. A chip's probability is looked up by the tag's index in the INTENTS order
// that GET /api/intents serves (the order of the answer's probabilities); while that list is missing the chip shows
// its tag name only, without the question tooltip. Server strings are rendered as React text nodes.

import { useId, useState, type CSSProperties } from "react";
import { formatProbability, formatTemperature, formatTheta } from "../format";
import { ChevronDownIcon, ChevronRightIcon, InfoIcon } from "../icons";
import Tooltip from "../Tooltip";
import TagPanel, { withQuestion, type TagLookup } from "./TagPanel";
import type { Answer, ClassifyResponse } from "./types";
import css from "./chat.module.css";

const NO_INTENT_HEADING = "No intent";
const NO_INTENT_BODY = "No tag reached its threshold θ for this message.";
const CONFIDENCE_LABEL = "Message confidence";
const CONFIDENCE_HINT = "Low means route to a human.";
const CONFIDENCE_INFO_LABEL = "How message confidence is computed";
const SHOW_ALL = "Show all 48 tags";

interface ColumnProps {
  name: "Laya" | "BERTurk";
  series: string;
  answer: Answer;
  tags: TagLookup | null;
  confidence_rule: string | null;
  open: boolean;
  onToggle: () => void;
}

function settingsLine(a: Answer): string {
  return (
    `T ${formatTemperature(a.temperature)} · θ ${formatTheta(a.theta_global)} global · ` +
    `${a.n_own_theta} tags with their own θ`
  );
}

function Column({ name, series, answer, tags, confidence_rule, open, onToggle }: ColumnProps) {
  const style = { "--series": series } as CSSProperties;
  const panelId = useId();
  return (
    <section className={css.column} style={style} aria-label={`${name} reply`}>
      <div className={css.titleRow}>
        <span className={css.swatch} aria-hidden="true" />
        <h3 className={css.columnTitle}>{name}</h3>
        <span className={css.caption}>{`fine-tuned, ${answer.seeds.length} seeds averaged`}</span>
      </div>
      <p className={css.settings}>{settingsLine(answer)}</p>
      {answer.predicted.length > 0 ? (
        <ul className={css.chips}>
          {answer.predicted.map((tag) => {
            const i = tags?.get(tag)?.index;
            return (
              <li key={tag} className={css.chipItem}>
                {withQuestion(
                  tags,
                  tag,
                  <span className={css.chip} tabIndex={0}>
                    <span className={css.chipTag}>{tag}</span>
                    {i !== undefined && (
                      <span className={css.chipValue}>{formatProbability(answer.probabilities[i])}</span>
                    )}
                  </span>,
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <div className={css.noIntent}>
          <h4 className={css.noIntentHeading}>{NO_INTENT_HEADING}</h4>
          <p className={css.noIntentBody}>{NO_INTENT_BODY}</p>
        </div>
      )}
      <div className={css.confidence}>
        <div className={css.confidenceRow}>
          <span className={css.confidenceLabel}>{CONFIDENCE_LABEL}</span>
          <span className={css.confidenceValue}>{formatProbability(answer.confidence)}</span>
          <span className={css.bar} aria-hidden="true">
            <span className={css.barFill} style={{ width: `${answer.confidence * 100}%` }} />
          </span>
          {confidence_rule !== null && (
            <Tooltip text={confidence_rule}>
              <button type="button" className={css.info} aria-label={CONFIDENCE_INFO_LABEL}>
                <InfoIcon />
              </button>
            </Tooltip>
          )}
        </div>
        <p className={css.hint}>{CONFIDENCE_HINT}</p>
      </div>
      <div className={css.disclosure}>
        <button type="button" className={css.disclosureTrigger} aria-expanded={open} aria-controls={panelId}
          onClick={onToggle}>
          {open ? <ChevronDownIcon /> : <ChevronRightIcon />}
          {SHOW_ALL}
        </button>
        <TagPanel id={panelId} answer={answer} series={series} tags={tags} open={open} />
      </div>
    </section>
  );
}

interface ReplyCardProps {
  reply: ClassifyResponse;
  tags: TagLookup | null;
  confidence_rule: string | null;
}

export default function ReplyCard({ reply, tags, confidence_rule }: ReplyCardProps) {
  const [open, setOpen] = useState(false);
  const toggle = (): void => setOpen((o) => !o);
  return (
    <div className={css.card}>
      <Column name="Laya" series="var(--series-laya)" answer={reply.laya} tags={tags}
        confidence_rule={confidence_rule} open={open} onToggle={toggle} />
      <Column name="BERTurk" series="var(--series-berturk)" answer={reply.berturk} tags={tags}
        confidence_rule={confidence_rule} open={open} onToggle={toggle} />
    </div>
  );
}
