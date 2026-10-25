// "Test results and verdict" (11-UI-SPEC.md "3. Test results and verdict", D-10), from GET /api/verdict, which serves
// results/final/verdict.json behind require_frozen and check_record together with report.py's three renders. In the
// UI-SPEC's order: the verdict hero (the verdict, rule and means lines as the frozen render gives them, then the
// recommendation line), the criteria table with status badges, the results-on-test table and its settings table, and
// the caveats. "View the frozen render" under the hero, the criteria and the results opens the exact render text.
// Every sentence, status word, number and caveat is a field of the view, shown verbatim; the UI writes no verdict
// sentence and computes nothing. A refused record shows its one-line refusal and no number from it.

import { Card, DataTable, FrozenRender, SourceCaption, StatusBadge, ViewGate, useView } from "./common";
import type { VerdictData } from "./types";
import css from "./dashboard.module.css";

function Hero({ v }: { v: VerdictData }) {
  return (
    <Card className={css.hero} style={{ borderLeftColor: `var(--tone-${v.tone}-fill)` }}>
      <p className={css.verdictLine}>{v.lines.verdict}</p>
      <p className={css.heroLine}>{v.lines.rule}</p>
      <p className={css.heroLine}>{v.lines.means}</p>
      <p className={css.heroRecommendation}>{v.lines.recommendation}</p>
      <FrozenRender renders={[v.renders.go, v.renders.recommendation]} />
    </Card>
  );
}

function Criteria({ v }: { v: VerdictData }) {
  return (
    <Card title="Criteria">
      <DataTable
        wrap
        header={["criterion", "status", "numbers"]}
        rows={v.criteria.map((c) => [c.criterion, <StatusBadge status={c.status} tone={c.tone} />, c.numbers])}
      />
      <FrozenRender renders={[v.renders.go]} />
    </Card>
  );
}

function Results({ v }: { v: VerdictData }) {
  return (
    <>
      <Card title="Results on test">
        <DataTable header={v.results.header} rows={v.results.rows} />
        <FrozenRender renders={[v.renders.final_table]} />
      </Card>
      <Card title="Settings">
        <DataTable header={v.settings.header} rows={v.settings.rows} />
      </Card>
    </>
  );
}

function Caveats({ v }: { v: VerdictData }) {
  return (
    <Card title="Caveats">
      <ul className={css.caveats}>
        {v.caveats.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </Card>
  );
}

export default function VerdictView() {
  const state = useView("verdict");
  return (
    <ViewGate state={state}>
      {(v) => (
        <>
          <SourceCaption sources={v.sources} />
          <Hero v={v} />
          <Criteria v={v} />
          <Results v={v} />
          <Caveats v={v} />
        </>
      )}
    </ViewGate>
  );
}
