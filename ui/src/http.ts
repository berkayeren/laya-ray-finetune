// Same-origin JSON requests to the demo backend (demo.py on 127.0.0.1). Every call resolves to a discriminated
// result and never throws:
//   {ok: true, data}                        the server answered 2xx with a JSON body other than null;
//   {ok: false, status: number, error, source, served}  the server answered an HTTP error; error is its one-line
//                                           "error" field (served true: the line ends in its own fix) and source its
//                                           "source" field (the record a view read), or null; a body without an
//                                           "error" string gives error "HTTP <status>" and served false; a 2xx whose
//                                           body is not JSON (or is null) is this too, with error
//                                           "HTTP <status> (not JSON)", source null and served false, so data is
//                                           never null;
//   {ok: false, status: null, error: ""}    the server did not answer (unreachable).
// Paths are same-origin only ("/api/..."), so the page never reaches another host.

export type Result<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; error: string; source: string | null; served: boolean }
  | { ok: false; status: null; error: "" };

const UNREACHABLE = { ok: false, status: null, error: "" } as const;

async function toResult<T>(response: Response): Promise<Result<T>> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (response.ok) {
    if (body === null) {
      const error = "HTTP " + response.status + " (not JSON)";
      return { ok: false, status: response.status, error, source: null, served: false };
    }
    return { ok: true, data: body as T };
  }
  const fields = body !== null && typeof body === "object" ? (body as { error?: unknown; source?: unknown }) : {};
  const served = typeof fields.error === "string";
  const error = typeof fields.error === "string" ? fields.error : "HTTP " + response.status;
  const source = typeof fields.source === "string" ? fields.source : null;
  return { ok: false, status: response.status, error, source, served };
}

export async function getJSON<T>(path: string): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(path, { method: "GET", headers: { Accept: "application/json" } });
  } catch {
    return UNREACHABLE;
  }
  return toResult<T>(response);
}

export async function postJSON<T>(path: string, body: unknown): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    return UNREACHABLE;
  }
  return toResult<T>(response);
}
