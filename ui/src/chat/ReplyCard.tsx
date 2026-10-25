// One reply card (D-04, D-05): Laya left, BERTurk right, always in that order. Each column shows the system's settings
// line, its predicted tags as chips (in the order of the answer's predicted list) or the No intent block, and its
// message confidence with an info icon whose tooltip is the frozen confidence rule, exactly as GET /api/intents serves
// it (confidence_rule; while it is missing the icon is omitted, and the UI writes no rule text of its own). Every
// number comes from the answer: the UI computes no threshold, average or confidence, and it never sorts or compares a
// probability with θ. A chip's probability is looked up by the tag's index in the INTENTS order that GET /api/intents
// serves (the order of the answer's probabilities); while that list is missing the chip shows its tag name only.
// Server strings are rendered as React text nodes.

import type { CSSProperties } from "react";
import { formatProbability, formatTemperature, formatTheta } from "../format";
import { InfoIcon } from "../icons";
import Tooltip from "../Tooltip";
import type { Answer, ClassifyResponse } from "./types";
import css from "./chat.module.css";

const NO_INTENT_HEADING = "No intent";
const NO_INTENT_BODY = "No tag reached its threshold θ for this message.";
const CONFIDENCE_LABEL = "Message confidence";
const CONFIDENCE_HINT = "Low means route to a human.";
const CONFIDENCE_INFO_LABEL = "How message confidence is computed";

interface ColumnProps {
  name: "Laya" | "BERTurk";
  series: string;
  answer: Answer;
  tagIndex: ReadonlyMap<string, number> | null;
  confidence_rule: string | null;
}

function settingsLine(a: Answer): string {
  return (
    `T ${formatTemperature(a.temperature)} · θ ${formatTheta(a.theta_global)} global · ` +
    `${a.n_own_theta} tags with their own θ`
  );
}

function Column({ name, series, answer, tagIndex, confidence_rule }: ColumnProps) {
  const style = { "--series": series } as CSSProperties;
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
            const i = tagIndex?.get(tag);
            return (
              <li key={tag} className={css.chip}>
                <span className={css.chipTag}>{tag}</span>
                {i !== undefined && <span className={css.chipValue}>{formatProbability(answer.probabilities[i])}</span>}
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
    </section>
  );
}

interface ReplyCardProps {
  reply: ClassifyResponse;
  tagIndex: ReadonlyMap<string, number> | null;
  confidence_rule: string | null;
}

export default function ReplyCard({ reply, tagIndex, confidence_rule }: ReplyCardProps) {
  return (
    <div className={css.card}>
      <Column name="Laya" series="var(--series-laya)" answer={reply.laya} tagIndex={tagIndex}
        confidence_rule={confidence_rule} />
      <Column name="BERTurk" series="var(--series-berturk)" answer={reply.berturk} tagIndex={tagIndex}
        confidence_rule={confidence_rule} />
    </div>
  );
}
