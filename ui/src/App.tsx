// The app shell (11-UI-SPEC.md "App shell"): the 56px header with the title, the tabs "Chat" and "Dashboard", the
// frozen-verdict pill and the note "Local demo · stores nothing", over the view of the current hash route.
//
// Routes (window.location.hash and the hashchange event, no router dependency): #/chat, the default, and
// #/dashboard/{training|selection|verdict|confidence}; #/dashboard alone means #/dashboard/training. Every navigation
// is a hash link, so the browser's back button works across them.
//
// The chat route renders ChatView with no props whatever the chat status: ChatView polls GET /api/status itself and
// owns the chat's loading, refused and unreachable states. It stays mounted (hidden) while a dashboard view is open,
// so switching tabs keeps the conversation; a reload still empties it (nothing is stored).
//
// The verdict pill reads GET /api/verdict once. It is hidden while the request is pending and when the server does not
// answer; when the record view is refused it reads "Verdict unavailable" (the verdict view shows the refusal); when it
// answers, it shows the status badge of the record's verdict with the server's tone, then the record's means text,
// both from the API, never typed here. It opens #/dashboard/verdict.

import { useEffect, useState } from "react";
import ChatView from "./chat/ChatView";
import Dashboard, { DEFAULT_VIEW, isViewKey, type ViewKey } from "./dashboard/Dashboard";
import { StatusBadge } from "./dashboard/common";
import type { VerdictData } from "./dashboard/types";
import { getJSON } from "./http";
import css from "./App.module.css";

const TITLE = "Laya vs BERTurk: Turkish banking intent demo";
const TITLE_SHORT = "Laya vs BERTurk";
const NOTE = "Local demo · stores nothing";
const VERDICT_UNAVAILABLE = "Verdict unavailable";
const VERDICT_HREF = "#/dashboard/verdict";

type Route = { page: "chat" } | { page: "dashboard"; view: ViewKey };

// #/chat (and anything unknown) -> the chat; #/dashboard[/view] -> that dashboard view
function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/");
  if (parts[0] !== "dashboard") {
    return { page: "chat" };
  }
  const view = parts[1] ?? "";
  return { page: "dashboard", view: isViewKey(view) ? view : DEFAULT_VIEW };
}

function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onHash = (): void => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

type Pill = { kind: "hidden" } | { kind: "unavailable" } | { kind: "ok"; verdict: string; tone: string; means: string };

function usePill(): Pill {
  const [pill, setPill] = useState<Pill>({ kind: "hidden" });
  useEffect(() => {
    let cancelled = false;
    void getJSON<VerdictData>("/api/verdict").then((r) => {
      if (cancelled) {
        return;
      }
      if (r.ok) {
        setPill({ kind: "ok", verdict: r.data.verdict, tone: r.data.tone, means: r.data.means });
      } else {
        setPill(r.status === null ? { kind: "hidden" } : { kind: "unavailable" });
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return pill;
}

function VerdictPill({ pill }: { pill: Pill }) {
  if (pill.kind === "hidden") {
    return null;
  }
  return (
    <a className={css.pill} href={VERDICT_HREF}>
      {pill.kind === "ok" ? (
        <>
          <StatusBadge status={pill.verdict} tone={pill.tone} />
          <span className={css.pillMeans}>{pill.means}</span>
        </>
      ) : (
        <span className={css.pillMeans}>{VERDICT_UNAVAILABLE}</span>
      )}
    </a>
  );
}

export default function App() {
  const route = useRoute();
  const pill = usePill();
  const chat = route.page === "chat";
  return (
    <>
      <header className={css.header}>
        <span className={css.title}>
          <span className={css.titleFull}>{TITLE}</span>
          <span className={css.titleShort}>{TITLE_SHORT}</span>
        </span>
        <nav className={css.tabs} aria-label="Views">
          <a className={css.tab} href="#/chat" aria-current={chat ? "page" : undefined}>
            Chat
          </a>
          <a className={css.tab} href="#/dashboard" aria-current={chat ? undefined : "page"}>
            Dashboard
          </a>
        </nav>
        <div className={css.end}>
          <VerdictPill pill={pill} />
          <span className={css.note}>{NOTE}</span>
        </div>
      </header>
      <main className={css.chat} hidden={!chat}>
        <ChatView />
      </main>
      {route.page === "dashboard" && <Dashboard view={route.view} />}
    </>
  );
}
