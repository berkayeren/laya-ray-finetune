// The dashboard frame (11-UI-SPEC.md "Dashboard view", D-10): the 240px sidebar with the four views in the UI-SPEC's
// order and the content column (max 1200px) holding the view of the current hash route. Every route goes through one
// table, VIEW_COMPONENTS: the val selection view (plan 11-11) and the verdict view (plan 11-10) are registered; plan
// 11-12 adds the training and the confidence views. A route without a component in the table shows the shared LoadingCard.
// Each view opens with its h1 (rendered here) and then its own one-line source caption (SourceCaption, from the
// sources field of the records it reads). Navigation is by hash links only, so the back button works.

import type { ComponentType } from "react";
import { LoadingCard } from "./common";
import SelectionView from "./SelectionView";
import VerdictView from "./VerdictView";
import css from "./dashboard.module.css";

export type ViewKey = "training" | "selection" | "verdict" | "confidence";

export const VIEWS: { key: ViewKey; label: string }[] = [
  { key: "training", label: "Training runs" },
  { key: "selection", label: "Val selection" },
  { key: "verdict", label: "Test results and verdict" },
  { key: "confidence", label: "Confidence, latency and few-shot" },
];

export const DEFAULT_VIEW: ViewKey = "training";

export function isViewKey(s: string): s is ViewKey {
  return VIEWS.some((v) => v.key === s);
}

export function viewHash(key: ViewKey): string {
  return "#/dashboard/" + key;
}

export const VIEW_COMPONENTS: Partial<Record<ViewKey, ComponentType>> = {
  selection: SelectionView,
  verdict: VerdictView,
};

export default function Dashboard({ view }: { view: ViewKey }) {
  const active = VIEWS.find((v) => v.key === view) ?? VIEWS[0];
  const View = VIEW_COMPONENTS[active.key];
  return (
    <div className={css.layout}>
      <nav className={css.sidebar} aria-label="Dashboard views">
        <ul className={css.navList}>
          {VIEWS.map((v) => (
            <li key={v.key}>
              <a
                className={css.navLink}
                href={viewHash(v.key)}
                aria-current={v.key === active.key ? "page" : undefined}
              >
                {v.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <main className={css.content}>
        <div className={css.inner}>
          <h1 className={css.h1}>{active.label}</h1>
          {View === undefined ? <LoadingCard /> : <View key={active.key} />}
        </div>
      </main>
    </div>
  );
}
